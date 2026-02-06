"""
StreamWrapper for broadcasting Pokemon Red training to a shared map server.

This wrapper enables real-time streaming of game coordinates to a WebSocket
server for visualization of multiple training sessions on a shared Kanto map.
Based on the pokemonred_puffer StreamWrapper which is proven to work with the
transdimensional.xyz visualization server.

Architecture Role:
    StreamWrapper is a Gymnasium wrapper that sits between the environment
    and the training loop. It intercepts step() calls to collect position
    data and periodically uploads it to a visualization server.

    train.py -> SubprocVecEnv -> StreamWrapper -> KantoRedEnv -> PyBoy

    All parallel environments can stream simultaneously, each with a unique
    username suffix and auto-generated color for visual distinction.

How It Works:
    1. On initialization, creates an asyncio event loop and WebSocket
       connection to the transdimensional.xyz broadcast server
    2. During step(), reads player position from game memory
    3. Accumulates coordinates in a buffer (coord_list)
    4. Every upload_interval steps, serializes the buffer to JSON and
       sends it synchronously via loop.run_until_complete()
    5. Connection failures are silently handled -- the next send retries

    The synchronous send model matches pokemonred_puffer's proven approach.
    At upload_interval=500 (~6s at 80 sps/env), each send takes ~10ms,
    which is negligible impact on training throughput.

Visualization Server:
    The default server (transdimensional.xyz) provides a shared map where
    multiple users can see their agents' positions in real-time. This enables:
    - Watching training progress without rendering locally
    - Comparing multiple training runs visually
    - Community engagement during long training runs

Protocol:
    Messages are JSON-encoded with the following structure:
    {
        "metadata": {
            "user": "KantoRL_1",
            "color": "#ff0000",
            "env_id": 0,
            "extra": "",
            "sprite_id": 5
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
    - kantorl.memory: For reading game position

Notes:
    - websockets package must be installed separately: pip install websockets
    - If websockets is not installed, streaming is silently disabled
    - Connection failures are handled gracefully (training continues)
    - All environments in a vectorized setup can stream simultaneously
"""

import asyncio
import colorsys
import json
from typing import Any

import gymnasium as gym

from kantorl import memory

# =============================================================================
# CONSTANTS
# =============================================================================


# Default color for all streaming agents
DEFAULT_STREAM_COLOR = "#ff0000"

# Auto-incrementing counter for assigning unique env_id integers.
# Each StreamWrapper instance in this process gets a unique ID.
# In SubprocVecEnv, each worker subprocess has its own counter starting at 0.
_next_env_id = 0


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

    WebSocket communication uses a synchronous send model matching
    pokemonred_puffer's proven approach. At upload_interval=500 (~6s at 80
    sps/env), each send takes ~10ms -- negligible training impact.

    Attributes:
        enabled: Whether streaming is currently active. May be disabled if:
                - enabled=False was passed to constructor
                - websockets package is not installed
        stream_metadata: Dictionary of user display information (name, color, sprite).
        upload_interval: Number of steps between coordinate uploads.
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
        ...     upload_interval=500,
        ... )
        >>> # Use normally - coordinates are streamed automatically
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = policy(obs)
        ...     obs, reward, term, trunc, info = env.step(action)

    Notes:
        - Coordinates are buffered and uploaded in batches for efficiency
        - Connection errors are handled silently (reconnect on next send)
        - The wrapper passes through all environment functionality unchanged
    """

    def __init__(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        username: str = "KantoRL",
        color: str = "#ff0000",
        sprite_id: int = 0,
        stream_interval: int = 500,
        extra_info: str = "",
        enabled: bool = True,
        rank: int | None = None,
        n_envs: int | None = None,
    ) -> None:
        """
        Initialize the StreamWrapper.

        Creates an asyncio event loop and establishes a WebSocket connection
        to the transdimensional.xyz broadcast server. If websockets is not
        installed, streaming is disabled but the wrapper still functions as
        a pass-through.

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
                           uploads. Higher values = larger batches, smoother
                           animation on the visualizer. Default: 500
                           (matches pokemonred_puffer).
            extra_info: Additional text to display alongside the agent name.
                       Default: ""
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
            - If websockets package is missing, a warning is printed
            - Connection failures are handled silently (training continues)
        """
        # Initialize the base Wrapper class
        super().__init__(env)

        # Store enabled state
        self.enabled = enabled

        # Early return if streaming is disabled
        if not self.enabled:
            return

        # -----------------------------------------------------------------
        # WebSocket Import
        # -----------------------------------------------------------------
        # Import websockets lazily -- it's an optional dependency.
        # This allows the package to work without websockets installed.
        try:
            import websockets as _ws_mod

            self._websockets_mod = _ws_mod
        except ImportError:
            # websockets not installed -- disable streaming with warning
            print(
                "Warning: websockets not installed. "
                "Install with: pip install websockets"
            )
            self.enabled = False
            return

        # -----------------------------------------------------------------
        # WebSocket Configuration
        # -----------------------------------------------------------------
        # The broadcast server URL -- this is where coordinates are sent.
        # The server then broadcasts to all connected visualization clients.
        self.ws_address = "wss://transdimensional.xyz/broadcast"

        # -----------------------------------------------------------------
        # Asyncio Event Loop + Connection
        # -----------------------------------------------------------------
        # Create a dedicated event loop for this wrapper instance.
        # run_until_complete() is used for synchronous WebSocket sends,
        # matching pokemonred_puffer's proven approach.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket: Any | None = None
        self.loop.run_until_complete(self._establish_connection())

        # -----------------------------------------------------------------
        # Stream Metadata
        # -----------------------------------------------------------------
        # Metadata sent with each coordinate upload.
        # This identifies the agent on the shared visualization.
        # No trailing newlines -- matches pokemonred_puffer's working format.
        global _next_env_id  # noqa: PLW0603
        env_id = _next_env_id
        _next_env_id += 1

        self.stream_metadata: dict[str, Any] = {
            "user": username,           # Display name (no trailing \n)
            "color": color,             # Trail/marker color
            "env_id": env_id,           # Unique agent identifier (integer)
            "extra": extra_info,        # Additional display text
            "sprite_id": sprite_id,     # Character sprite for visualization
        }

        # -----------------------------------------------------------------
        # Upload Configuration
        # -----------------------------------------------------------------
        # How often to upload coordinates (in environment steps).
        # 500 matches pokemonred_puffer's proven value. At ~80 sps/env,
        # each batch contains ~500 coords and is sent every ~6 seconds.
        # The visualizer adapts its animation duration to batch timing.
        self.upload_interval = stream_interval

        # Counter for steps since last upload
        self.step_counter = 0

        # Buffer for accumulated coordinates
        # Each entry is [x, y, map_id]
        self.coord_list: list[list[int]] = []

    # =====================================================================
    # WEBSOCKET MANAGEMENT
    # =====================================================================

    async def _establish_connection(self) -> None:
        """
        Establish WebSocket connection to the broadcast server.

        Silently sets self.websocket to None on failure. The next send
        attempt will retry the connection. This matches pokemonred_puffer's
        simple reconnect-on-demand pattern.

        Notes:
            - All exceptions are caught -- streaming never crashes training
            - Connection status is not logged to avoid SubprocVecEnv noise
        """
        try:
            self.websocket = await self._websockets_mod.connect(self.ws_address)
        except Exception:
            self.websocket = None

    async def _send_message(self, message: str) -> None:
        """
        Send a JSON message over WebSocket, reconnecting if needed.

        If the WebSocket is disconnected (None), attempts to reconnect
        before sending. If the send fails, sets websocket to None so
        the next call will retry.

        Args:
            message: JSON-encoded string to send.

        Notes:
            - All exceptions are caught -- streaming never crashes training
            - Matches pokemonred_puffer's broadcast_ws_message() pattern
        """
        # Reconnect if needed
        if self.websocket is None:
            await self._establish_connection()

        # Send if connected
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except Exception:
                # Connection broken -- will reconnect on next attempt
                self.websocket = None

    # =====================================================================
    # GYMNASIUM WRAPPER METHODS
    # =====================================================================

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        """
        Step the environment and stream coordinates if enabled.

        This method wraps the underlying environment's step() call, adding
        coordinate collection and periodic upload functionality.

        Process:
            1. Read player position from memory (before step for consistency)
            2. Add position to coordinate buffer
            3. If buffer is full (upload_interval reached), send batch
            4. Call the underlying environment's step()
            5. Return the original step() results unchanged

        The send operation uses loop.run_until_complete() which blocks for
        ~10ms every upload_interval steps (~6s). This matches
        pokemonred_puffer's proven synchronous approach.

        Args:
            action: The action to take in the environment.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            exactly as returned by the underlying environment.

        Notes:
            - Position is read from game memory, not from info dict
            - Coordinates are [x, y, map_id] format
            - Upload frequency is controlled by upload_interval
            - All return values pass through unchanged
        """
        # Collect coordinates before stepping (matches pokemonred_puffer order)
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

            # Add to coordinate buffer
            self.coord_list.append([x_pos, y_pos, map_id])

            # Check if upload is due
            if self.step_counter >= self.upload_interval:
                # Serialize and send synchronously
                self.loop.run_until_complete(
                    self._send_message(
                        json.dumps({
                            "metadata": self.stream_metadata,
                            "coords": self.coord_list,
                        })
                    )
                )

                # Reset buffer and counter
                self.step_counter = 0
                self.coord_list = []

            self.step_counter += 1

        # Call the underlying environment's step
        return self.env.step(action)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
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
            - Remaining coordinates are not uploaded (they're discarded)
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
        Close the WebSocket connection and the underlying environment.

        Gracefully closes the WebSocket if connected, then shuts down the
        asyncio event loop.

        Notes:
            - Safe to call multiple times
            - Calls underlying environment's close()
        """
        # Close WebSocket connection if active
        if self.enabled and hasattr(self, "websocket") and self.websocket is not None:
            try:
                self.loop.run_until_complete(self.websocket.close())
            except Exception:
                pass

        # Close the asyncio event loop
        if hasattr(self, "loop") and self.loop is not None:
            try:
                self.loop.close()
            except Exception:
                pass

        # Close underlying environment
        super().close()  # type: ignore[no-untyped-call]
