"""
StreamWrapper for broadcasting Pokemon Red training to a shared map server.

This wrapper enables real-time streaming of game coordinates to a WebSocket
server for visualization of multiple training sessions on a shared Kanto map.
This is based on the PokemonRedExperiments StreamWrapper implementation and
enables collaborative visualization of multiple agents training simultaneously.

Architecture Role:
    StreamWrapper is a Gymnasium wrapper that sits between the environment
    and the training loop. It intercepts step() calls to collect position
    data and periodically uploads it to a visualization server.

    train.py -> SubprocVecEnv -> StreamWrapper -> KantoRedEnv -> PyBoy

    All parallel environments can stream simultaneously, each with a unique
    username suffix and auto-generated color for visual distinction.

How It Works:
    1. On initialization, spawns a daemon sender thread with its own event
       loop and WebSocket connection
    2. During step(), reads player position from game memory
    3. Accumulates coordinates in a buffer (main thread only)
    4. Every stream_interval steps, serializes the buffer to JSON and
       enqueues it via a bounded queue (non-blocking put_nowait)
    5. The sender thread drains the queue and sends messages over WebSocket
    6. If the queue is full (slow network), messages are silently dropped
       -- stale coordinates have no visualization value

Threading Model:
    Main Thread (env.step)          Sender Daemon Thread
    ----------------------          --------------------
    collect coordinates             own asyncio event loop
    json.dumps(payload)             WebSocket connection
    queue.put_nowait(msg) -------> queue.get(timeout=0.5)
      (drops if full)                 websocket.send(msg)
                                      reconnect on failure
    close() -> shutdown.set()  --> websocket.close(), loop.close()

    Thread safety is guaranteed by design:
    - queue.Queue is inherently thread-safe
    - threading.Event is inherently thread-safe
    - coord_list and step_counter are main-thread-only
    - WebSocket + event loop are sender-thread-only (exclusive ownership)

Visualization Server:
    The default server (transdimensional.xyz) provides a shared map where
    multiple users can see their agents' positions in real-time. This enables:
    - Watching training progress without rendering locally
    - Comparing multiple training runs visually
    - Community engagement during long training runs

Protocol:
    Messages are JSON-encoded with the following structure, matching the
    transdimensional.xyz server convention (trailing newlines on strings):
    {
        "metadata": {
            "user": "username\\n",
            "color": "#ff0000",
            "env_id": "a1b2c3d4:1\\n",
            "extra": "\\n"
        },
        "coords": [[x, y, map_id], [x, y, map_id], ...]
    }

Usage:
    # Streaming is typically enabled via CLI flags
    kantorl train pokemon_red.gb --stream --stream-user "my_agent"

    # Or programmatically
    from kantorl.stream_wrapper import StreamWrapper

    env = KantoRedEnv(config=config)
    wrapped_env = StreamWrapper(
        env,
        username="my_agent",
        color="#ff0000",
        sprite_id=5,
    )

Dependencies:
    - gymnasium: For the Wrapper base class
    - websockets: For WebSocket communication (optional, installed separately)
    - json: For message serialization
    - queue: For thread-safe bounded message passing
    - threading: For the background sender daemon thread
    - kantorl.memory: For reading game position

Notes:
    - websockets package must be installed separately: pip install websockets
    - If websockets is not installed, streaming is silently disabled
    - Connection failures are handled gracefully (training continues)
    - All environments in a vectorized setup can stream simultaneously
    - WebSocket uploads are fully non-blocking -- the sender thread owns
      the connection and the main thread never waits on network I/O
"""

import asyncio
import colorsys
import json
import queue
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from kantorl import memory


# =============================================================================
# CONSTANTS
# =============================================================================


# Default color for all streaming agents
DEFAULT_STREAM_COLOR = "#ff0000"

# Maximum number of queued messages waiting to be sent.
# Small on purpose -- stale coordinates are worthless for real-time visualization.
# If the sender thread can't keep up, new messages replace old ones via silent drop.
_SEND_QUEUE_MAXSIZE = 2

# Unique identifier for this process/run, shared by all StreamWrapper instances.
# Used in the env_id metadata field so the server can distinguish agents across runs.
# Format matches PufferLib convention: 8-char hex prefix from UUID4.
_RUN_ID = uuid.uuid4().hex[:8]


# =============================================================================
# COLOR GENERATION
# =============================================================================


def _generate_agent_color(rank: int, n_envs: int) -> str:
    """
    Generate a unique color for each agent using HSL hue rotation.

    Distributes hues evenly around the color wheel so that N agents get
    maximally distinct colors. Uses high saturation and mid lightness for
    vibrant, easily distinguishable markers on the shared map.

    Args:
        rank: This agent's index (0 to n_envs-1).
        n_envs: Total number of parallel environments.

    Returns:
        Hex color string in "#RRGGBB" format.

    Example:
        >>> _generate_agent_color(0, 4)  # Red-ish
        '#e51a1a'
        >>> _generate_agent_color(1, 4)  # Green-ish
        '#1ae51a'
        >>> _generate_agent_color(2, 4)  # Blue-ish
        '#1a1ae5'
    """
    # Evenly space hues around the color wheel [0.0, 1.0)
    hue = rank / max(n_envs, 1)

    # colorsys.hls_to_rgb expects (hue, lightness, saturation)
    # lightness=0.5 gives pure colors, saturation=0.9 keeps them vivid
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.9)

    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


# =============================================================================
# SENDER THREAD FUNCTION
# =============================================================================


def _sender_thread_fn(
    ws_address: str,
    send_queue: "queue.Queue[Optional[str]]",
    shutdown_event: threading.Event,
    websockets_mod: Any,
    label: str,
) -> None:
    """
    Background thread that owns a WebSocket connection and sends queued messages.

    This function runs in a daemon thread and is the exclusive owner of both the
    asyncio event loop and the WebSocket connection. The main thread communicates
    with it solely via the thread-safe ``send_queue``.

    Lifecycle:
        1. Creates a new asyncio event loop (safe in a non-main thread)
        2. Connects to the WebSocket server
        3. Loops: dequeue message -> send via WebSocket
        4. On connection error, reconnects with exponential backoff (1s -> 30s)
        5. Exits when ``shutdown_event`` is set or a ``None`` sentinel is received

    Args:
        ws_address: WebSocket URL to connect to (e.g., "wss://transdimensional.xyz/broadcast").
        send_queue: Thread-safe bounded queue of JSON-encoded message strings.
            A ``None`` value acts as a shutdown sentinel.
        shutdown_event: Set by the main thread to signal graceful shutdown.
        websockets_mod: The imported ``websockets`` module (passed to avoid
            re-importing in the thread).
        label: Human-readable label for log messages (e.g., "KantoRL_3").

    Notes:
        - This function never raises -- all exceptions are caught and logged.
        - The thread is created as a daemon, so it dies automatically if the
          process exits without calling close().
        - Exponential backoff caps at 30 seconds to avoid long silent periods.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws: Optional[Any] = None
    backoff = 1.0  # seconds, doubles on failure up to 30s cap

    async def _connect() -> Any:
        """Establish WebSocket connection, returns connection object or None."""
        try:
            conn = await websockets_mod.connect(ws_address)
            print(f"StreamWrapper [{label}]: Connected to {ws_address}")
            return conn
        except Exception as e:
            print(f"StreamWrapper [{label}]: Connection failed: {e}")
            return None

    async def _run() -> None:
        """Main send loop: dequeue messages and send them over WebSocket."""
        nonlocal ws, backoff

        ws = await _connect()

        while not shutdown_event.is_set():
            # -----------------------------------------------------------------
            # Dequeue next message (blocks up to 0.5s so we can check shutdown)
            # -----------------------------------------------------------------
            try:
                msg = send_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # None sentinel means graceful shutdown
            if msg is None:
                break

            # -----------------------------------------------------------------
            # Send the message, reconnecting on failure
            # -----------------------------------------------------------------
            if ws is None:
                # Not connected -- try to reconnect
                ws = await _connect()
                if ws is not None:
                    backoff = 1.0  # reset backoff on success
                else:
                    # Still can't connect -- drop message, wait with backoff
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue

            try:
                await ws.send(msg)
                backoff = 1.0  # reset backoff on successful send
            except Exception:
                # Connection broken -- close and set to None for reconnect
                try:
                    await ws.close()
                except Exception:
                    pass
                ws = None
                # Message is lost -- acceptable for visualization data

        # -------------------------------------------------------------------
        # Cleanup: close WebSocket gracefully
        # -------------------------------------------------------------------
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    try:
        loop.run_until_complete(_run())
    except Exception:
        pass  # Thread must never propagate exceptions
    finally:
        loop.close()


# =============================================================================
# STREAM WRAPPER CLASS
# =============================================================================


class StreamWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """
    Gymnasium wrapper that streams game coordinates to a shared map server.

    This wrapper broadcasts player position and metadata to a WebSocket server,
    enabling real-time visualization of multiple Pokemon Red training sessions
    on a shared Kanto region map.

    The wrapper is designed to be minimally invasive -- it only reads position
    data and doesn't modify the observation, reward, or game state in any way.
    If streaming fails or is disabled, the environment continues to work normally.

    WebSocket communication is fully non-blocking: a background daemon thread
    owns the connection and event loop, while the main thread enqueues messages
    through a bounded ``queue.Queue``. If the queue is full (slow network),
    stale coordinate data is silently dropped with zero training impact.

    Attributes:
        enabled: Whether streaming is currently active. May be disabled if:
                - enabled=False was passed to constructor
                - websockets package is not installed
        stream_metadata: Dictionary of user display information (name, color, sprite).
        stream_interval: Number of steps between coordinate uploads.
        step_counter: Counter tracking steps since last upload.
        coord_list: Buffer of accumulated coordinates to upload.

    Example:
        >>> env = KantoRedEnv(config=config)
        >>> # Wrap with streaming enabled
        >>> env = StreamWrapper(
        ...     env,
        ...     username="agent_1",
        ...     color="#00ff00",
        ...     sprite_id=3,
        ...     stream_interval=10,
        ... )
        >>> # Use normally - coordinates are streamed automatically
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = policy(obs)
        ...     obs, reward, term, trunc, info = env.step(action)

    Notes:
        - A background daemon thread handles all WebSocket I/O
        - Coordinates are buffered and uploaded in batches for efficiency
        - Connection errors are handled in the background thread (reconnect with backoff)
        - The wrapper passes through all environment functionality unchanged
    """

    def __init__(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        username: str = "KantoRL",
        color: str = "#ff0000",
        sprite_id: int = 0,
        stream_interval: int = 300,
        extra_info: str = "",
        enabled: bool = True,
        rank: int | None = None,
        n_envs: int | None = None,
    ) -> None:
        """
        Initialize the StreamWrapper.

        Sets up a background sender thread and configures streaming metadata.
        If websockets is not installed, streaming is disabled but the wrapper
        still functions as a pass-through.

        When rank and n_envs are provided and the color is the default,
        an auto-generated color is assigned using HSL hue rotation so that
        each parallel agent gets a visually distinct marker on the shared map.

        Args:
            env: The Gymnasium environment to wrap. Must have a ``pyboy``
                attribute for reading game memory (i.e., KantoRedEnv).
            username: Display name for this agent on the shared map.
                     Visible to other users viewing the visualization.
                     Default: "KantoRL"
            color: Hex color code for the agent's trail/marker on the map.
                  Format: "#RRGGBB" (e.g., "#0033ff" for blue).
                  Default: "#ff0000" (red).
            sprite_id: Character sprite ID for visualization (0-50).
                      Different sprites show different Pokemon trainer appearances.
                      Default: 0
            stream_interval: Number of environment steps between coordinate
                           uploads. Lower values = more frequent updates but
                           more network traffic. Default: 300
            extra_info: Additional text to display alongside the agent name.
                       Currently unused but reserved for future features.
            enabled: Whether to enable streaming. Set to False to create
                    the wrapper without attempting to connect. Useful for
                    conditional streaming in vectorized environments.
                    Default: True
            rank: This environment's index (0 to n_envs-1). When provided
                 along with n_envs, enables auto-color generation for
                 distinct per-agent colors.
            n_envs: Total number of parallel environments. Used with rank
                   to compute evenly-spaced colors.

        Notes:
            - A daemon sender thread is spawned that owns the WebSocket connection
            - If websockets package is missing, a warning is printed
            - Connection failures are handled in the background thread
        """
        # Initialize the base Wrapper class
        super().__init__(env)

        # Store enabled state
        self.enabled = enabled

        # Early return if streaming is disabled
        if not self.enabled:
            return

        # ---------------------------------------------------------------------
        # WebSocket Import
        # ---------------------------------------------------------------------
        # Import websockets lazily -- it's an optional dependency.
        # This allows the package to work without websockets installed.
        try:
            import websockets

            websockets_mod = websockets
        except ImportError:
            # websockets not installed -- disable streaming with warning
            print("Warning: websockets not installed. Install with: pip install websockets")
            self.enabled = False
            return

        # ---------------------------------------------------------------------
        # WebSocket Configuration
        # ---------------------------------------------------------------------
        # The broadcast server URL -- this is where coordinates are sent.
        # The server then broadcasts to all connected visualization clients.
        ws_address = "wss://transdimensional.xyz/broadcast"

        # ---------------------------------------------------------------------
        # Stream Metadata
        # ---------------------------------------------------------------------
        # Metadata sent with each coordinate upload.
        # This identifies the agent on the shared visualization.
        # String values have trailing "\n" to match the transdimensional.xyz
        # server protocol convention (as used by PufferLib and others).
        # env_id uniquely identifies this agent across runs and workers.
        rank_str = str(rank + 1) if rank is not None else "0"
        self.stream_metadata: Dict[str, Any] = {
            "user": username + "\n",                      # Display name
            "color": color,                               # Trail/marker color
            "env_id": f"{_RUN_ID}:{rank_str}\n",          # Unique agent identifier
            "extra": extra_info + "\n",                    # Additional display text
        }

        # ---------------------------------------------------------------------
        # Upload Configuration
        # ---------------------------------------------------------------------
        # How often to upload coordinates (in environment steps)
        self.stream_interval = stream_interval

        # Counter for steps since last upload
        self.step_counter = 0

        # Buffer for accumulated coordinates
        # Each entry is [x, y, map_id]
        self.coord_list: list[list[int]] = []

        # ---------------------------------------------------------------------
        # Background Sender Thread
        # ---------------------------------------------------------------------
        # Bounded queue for passing serialized JSON messages to the sender thread.
        # Small maxsize because stale coordinates are worthless -- if the sender
        # can't keep up, we silently drop rather than accumulate a backlog.
        self._send_queue: queue.Queue[Optional[str]] = queue.Queue(
            maxsize=_SEND_QUEUE_MAXSIZE,
        )

        # Event to signal the sender thread to shut down gracefully
        self._shutdown_event = threading.Event()

        # Human-readable label for log messages
        label = username

        # Spawn the sender daemon thread.
        # daemon=True ensures it dies automatically if the process exits
        # without calling close() (e.g., SubprocVecEnv worker crash).
        self._sender_thread = threading.Thread(
            target=_sender_thread_fn,
            args=(ws_address, self._send_queue, self._shutdown_event, websockets_mod, label),
            name=f"stream-sender-{username}",
            daemon=True,
        )
        self._sender_thread.start()

    # =========================================================================
    # GYMNASIUM WRAPPER METHODS
    # =========================================================================

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Step the environment and stream coordinates if enabled.

        This method wraps the underlying environment's step() call, adding
        coordinate collection and periodic upload functionality.

        Process:
            1. Call the underlying environment's step()
            2. If streaming is enabled, read player position from memory
            3. Add position to coordinate buffer
            4. If buffer is full (stream_interval reached), serialize and enqueue
            5. Return the original step() results unchanged

        The enqueue operation is non-blocking: ``put_nowait()`` either succeeds
        instantly or raises ``queue.Full``, which is silently caught. This
        guarantees zero latency impact on the training loop.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            exactly as returned by the underlying environment.

        Notes:
            - Position is read from game memory, not from info dict
            - Coordinates are [x, y, map_id] format
            - Upload frequency is controlled by stream_interval
            - All return values pass through unchanged
        """
        # Call the underlying environment's step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Stream coordinates if enabled and pyboy is available.
        # Use self.unwrapped to bypass any intermediate wrappers (e.g.
        # CurriculumWrapper) -- @property descriptors on KantoRedEnv are
        # not forwarded by gym.Wrapper.__getattr__ in Gymnasium 1.x.
        base_env = self.unwrapped
        if self.enabled and hasattr(base_env, "pyboy"):
            # Get current position from game memory
            # These addresses are defined in memory.py
            pyboy = base_env.pyboy
            x_pos = memory.read_byte(pyboy, memory.ADDR_PLAYER_X)
            y_pos = memory.read_byte(pyboy, memory.ADDR_PLAYER_Y)
            map_id = memory.read_byte(pyboy, memory.ADDR_MAP_ID)

            # Filter out uninitialized/invalid (0, 0, 0) positions.
            # These occur before the game state is fully loaded and would
            # place the agent at the map origin on the visualization.
            if x_pos != 0 or y_pos != 0 or map_id != 0:
                # Add to coordinate buffer
                # Format: [x, y, map_id]
                self.coord_list.append([x_pos, y_pos, map_id])

            # Increment step counter and check if upload is due
            self.step_counter += 1
            if self.step_counter >= self.stream_interval:
                # Serialize and enqueue for the background sender thread
                self._enqueue_coordinates()

                # Reset buffer and counter
                self.step_counter = 0
                self.coord_list = []

        # Return original step results unchanged
        return obs, reward, terminated, truncated, info

    def _enqueue_coordinates(self) -> None:
        """
        Serialize accumulated coordinates and enqueue for background sending.

        Packages the coordinate buffer with metadata into a JSON string and
        attempts a non-blocking put onto the sender queue. If the queue is
        full (sender thread is behind), the message is silently dropped --
        stale coordinates have no value for real-time visualization.

        Message Format:
            {
                "metadata": {
                    "user": "username\\n",
                    "color": "#ff0000",
                    "env_id": "a1b2c3d4:1\\n",
                    "extra": "\\n"
                },
                "coords": [[x, y, map_id], ...]
            }

        Notes:
            - No-op if coordinate buffer is empty
            - Non-blocking: never waits on network I/O
            - queue.Full is expected under slow network conditions
        """
        # Don't upload if buffer is empty
        if not self.coord_list:
            return

        # Create JSON-encoded message payload
        message = json.dumps({
            "metadata": self.stream_metadata,
            "coords": self.coord_list,
        })

        # Non-blocking enqueue -- drop silently if the sender can't keep up
        try:
            self._send_queue.put_nowait(message)
        except queue.Full:
            pass  # Stale coordinates are worthless; dropping is correct

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and clear the coordinate buffer.

        Calls the underlying environment's reset() and clears any accumulated
        coordinates to start fresh for the new episode.

        Args:
            **kwargs: Keyword arguments passed to the underlying reset().

        Returns:
            Tuple of (observation, info) from the underlying environment.

        Notes:
            - Coordinate buffer is cleared on reset
            - Step counter is reset to 0
            - Does not upload remaining coordinates (they're discarded)
        """
        # Call underlying environment reset
        obs, info = self.env.reset(**kwargs)

        # Clear streaming state for new episode
        if self.enabled:
            self.coord_list = []
            self.step_counter = 0

        return obs, info

    def close(self) -> None:
        """
        Signal the sender thread to shut down and close the underlying environment.

        Sets the shutdown event so the sender thread exits its loop, then joins
        the thread with a 5-second timeout. If the thread doesn't exit in time
        (e.g., stuck in a reconnect backoff), it's abandoned as a daemon thread
        and will die when the process exits.

        Notes:
            - Safe to call multiple times
            - Calls underlying environment's close()
            - Handles cases where the sender thread was never started
        """
        # Signal the sender thread to shut down
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()

            # Send a None sentinel to unblock queue.get() immediately
            try:
                self._send_queue.put_nowait(None)
            except queue.Full:
                pass  # Shutdown event will also cause thread to exit

            # Wait for the sender thread to finish (bounded wait)
            if hasattr(self, "_sender_thread"):
                self._sender_thread.join(timeout=5.0)

        # Close underlying environment
        super().close()
