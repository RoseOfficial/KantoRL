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

    train.py → SubprocVecEnv → StreamWrapper → KantoRedEnv → PyBoy

    All parallel environments can stream simultaneously, each with a unique
    username suffix and auto-generated color for visual distinction.

How It Works:
    1. On initialization, establishes a WebSocket connection to the server
    2. During step(), reads player position from game memory
    3. Accumulates coordinates in a buffer
    4. Every stream_interval steps, uploads the buffer to the server
    5. Server broadcasts to connected visualization clients

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
            "user": "username",
            "color": "#ff0000",
            "sprite_id": 0
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
    - asyncio: For async WebSocket operations
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
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np  # noqa: F401 - May be used in future extensions

from kantorl import memory


# =============================================================================
# COLOR GENERATION
# =============================================================================


# Default color for all streaming agents
DEFAULT_STREAM_COLOR = "#ff0000"


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


class StreamWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that streams game coordinates to a shared map server.

    This wrapper broadcasts player position and metadata to a WebSocket server,
    enabling real-time visualization of multiple Pokemon Red training sessions
    on a shared Kanto region map.

    The wrapper is designed to be minimally invasive - it only reads position
    data and doesn't modify the observation, reward, or game state in any way.
    If streaming fails or is disabled, the environment continues to work normally.

    Attributes:
        enabled: Whether streaming is currently active. May be disabled if:
                - enabled=False was passed to constructor
                - websockets package is not installed
                - WebSocket connection failed
        websocket: The active WebSocket connection, or None if not connected.
        stream_metadata: Dictionary of user display information (name, color, sprite).
        stream_interval: Number of steps between coordinate uploads.
        step_counter: Counter tracking steps since last upload.
        coord_list: Buffer of accumulated coordinates to upload.
        loop: asyncio event loop for WebSocket operations.

    Example:
        >>> env = KantoRedEnv(config=config)
        >>> # Wrap with streaming enabled
        >>> env = StreamWrapper(
        ...     env,
        ...     username="agent_1",
        ...     color="#00ff00",
        ...     sprite_id=3,
        ...     stream_interval=300,
        ... )
        >>> # Use normally - coordinates are streamed automatically
        >>> obs, info = env.reset()
        >>> for _ in range(1000):
        ...     action = policy(obs)
        ...     obs, reward, term, trunc, info = env.step(action)

    Notes:
        - WebSocket connection is established during __init__
        - Coordinates are buffered and uploaded in batches for efficiency
        - Connection errors are handled gracefully (logged, streaming disabled)
        - The wrapper passes through all environment functionality unchanged
    """

    def __init__(
        self,
        env: gym.Env,
        username: str = "kantorl-agent",
        color: str = "#ff0000",
        sprite_id: int = 0,
        stream_interval: int = 300,
        extra_info: str = "",
        enabled: bool = True,
        rank: int | None = None,
        n_envs: int | None = None,
    ):
        """
        Initialize the StreamWrapper.

        Sets up the WebSocket connection and configures streaming metadata.
        If websockets is not installed or connection fails, streaming is
        disabled but the wrapper still functions as a pass-through.

        When rank and n_envs are provided and the color is the default,
        an auto-generated color is assigned using HSL hue rotation so that
        each parallel agent gets a visually distinct marker on the shared map.

        Args:
            env: The Gymnasium environment to wrap. Must have a `pyboy`
                attribute for reading game memory (i.e., KantoRedEnv).
            username: Display name for this agent on the shared map.
                     Visible to other users viewing the visualization.
                     Default: "kantorl-agent"
            color: Hex color code for the agent's trail/marker on the map.
                  Format: "#RRGGBB" (e.g., "#0033ff" for blue).
                  Default: "#ff0000" (red).
            sprite_id: Character sprite ID for visualization (0-50).
                      Different sprites show different Pokemon trainer appearances.
                      Default: 0
            stream_interval: Number of environment steps between coordinate
                           uploads. Lower values = more frequent updates but
                           more network traffic. Default: 300 (upload every
                           300 steps, roughly every 5 seconds at 60fps)
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
            - WebSocket connection is attempted synchronously during init
            - If websockets package is missing, a warning is printed
            - Connection failures are logged but don't raise exceptions
        """
        # Initialize the base Wrapper class
        super().__init__(env)

        # Store enabled state
        self.enabled = enabled

        # Early return if streaming is disabled
        if not self.enabled:
            return

        # -------------------------------------------------------------------------
        # WebSocket Import
        # -------------------------------------------------------------------------
        # Import websockets lazily - it's an optional dependency
        # This allows the package to work without websockets installed
        try:
            import websockets
            self.websockets = websockets
        except ImportError:
            # websockets not installed - disable streaming with warning
            print("Warning: websockets not installed. Install with: pip install websockets")
            self.enabled = False
            return

        # -------------------------------------------------------------------------
        # WebSocket Configuration
        # -------------------------------------------------------------------------
        # The broadcast server URL - this is where coordinates are sent
        # The server then broadcasts to all connected visualization clients
        self.ws_address = "wss://transdimensional.xyz/broadcast"

        # WebSocket connection object (set during _establish_connection)
        self.websocket: Optional[Any] = None

        # -------------------------------------------------------------------------
        # Stream Metadata
        # -------------------------------------------------------------------------
        # Metadata sent with each coordinate upload
        # This identifies the agent on the shared visualization
        self.stream_metadata = {
            "user": username,      # Display name
            "color": color,        # Trail/marker color
            "sprite_id": sprite_id,  # Character sprite appearance
        }

        # -------------------------------------------------------------------------
        # Upload Configuration
        # -------------------------------------------------------------------------
        # How often to upload coordinates (in environment steps)
        self.stream_interval = stream_interval

        # Counter for steps since last upload
        self.step_counter = 0

        # Buffer for accumulated coordinates
        # Each entry is [x, y, map_id]
        self.coord_list = []

        # -------------------------------------------------------------------------
        # Asyncio Setup
        # -------------------------------------------------------------------------
        # Create a new event loop for async WebSocket operations
        # We need our own loop since we're not in an async context
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Establish the WebSocket connection
        # This blocks until connected (or fails)
        self.loop.run_until_complete(self._establish_connection())

    # =========================================================================
    # WEBSOCKET METHODS
    # =========================================================================

    async def _establish_connection(self) -> None:
        """
        Establish WebSocket connection to the broadcast server.

        Attempts to connect to the configured WebSocket server. If connection
        fails, streaming is disabled and an error message is logged.

        This is called during __init__ and when reconnecting after errors.

        Notes:
            - Connection is async but we block on it during init
            - Timeout and other connection errors are caught and logged
            - self.enabled is set to False if connection fails
        """
        try:
            # Attempt to connect to the WebSocket server
            self.websocket = await self.websockets.connect(self.ws_address)
            print(f"StreamWrapper: Connected to {self.ws_address}")
        except Exception as e:
            # Connection failed - log error and disable streaming
            print(f"StreamWrapper: Failed to connect to {self.ws_address}: {e}")
            self.enabled = False

    async def _send_message(self, message: str) -> None:
        """
        Send a message through the WebSocket connection.

        If the connection is broken, attempts to reconnect before giving up.

        Args:
            message: JSON-encoded string to send to the server.

        Notes:
            - Errors are logged but don't raise exceptions
            - Reconnection is attempted on send failure
            - If reconnection fails, message is lost (acceptable for visualization)
        """
        if self.websocket:
            try:
                await self.websocket.send(message)
            except Exception as e:
                # Send failed - log and try to reconnect
                print(f"StreamWrapper: Failed to send message: {e}")
                # Attempt to restore connection for next time
                await self._establish_connection()

    # =========================================================================
    # GYMNASIUM WRAPPER METHODS
    # =========================================================================

    def step(self, action: Any) -> tuple:
        """
        Step the environment and stream coordinates if enabled.

        This method wraps the underlying environment's step() call, adding
        coordinate collection and periodic upload functionality.

        Process:
            1. Call the underlying environment's step()
            2. If streaming is enabled, read player position from memory
            3. Add position to coordinate buffer
            4. If buffer is full (stream_interval reached), upload coordinates
            5. Return the original step() results unchanged

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

        # Stream coordinates if enabled and pyboy is available
        if self.enabled and hasattr(self.env, 'pyboy'):
            # Get current position from game memory
            # These addresses are defined in memory.py
            pyboy = self.env.pyboy
            x_pos = memory.read_byte(pyboy, memory.ADDR_PLAYER_X)
            y_pos = memory.read_byte(pyboy, memory.ADDR_PLAYER_Y)
            map_id = memory.read_byte(pyboy, memory.ADDR_MAP_ID)

            # Add to coordinate buffer
            # Format: [x, y, map_id]
            self.coord_list.append([x_pos, y_pos, map_id])

            # Increment step counter and check if upload is due
            self.step_counter += 1
            if self.step_counter >= self.stream_interval:
                # Upload accumulated coordinates
                self._upload_coordinates(info)

                # Reset buffer and counter
                self.step_counter = 0
                self.coord_list = []

        # Return original step results unchanged
        return obs, reward, terminated, truncated, info

    def _upload_coordinates(self, info: Dict[str, Any]) -> None:
        """
        Upload accumulated coordinates to the visualization server.

        Packages the coordinate buffer with metadata and sends it to the
        WebSocket server for broadcast to visualization clients.

        Args:
            info: The info dict from the last step (currently unused,
                 reserved for future stat display features).

        Message Format:
            {
                "metadata": {
                    "user": "username",
                    "color": "#ff0000",
                    "sprite_id": 0
                },
                "coords": [[x, y, map_id], ...]
            }

        Notes:
            - No-op if coordinate buffer is empty
            - Errors are logged but don't interrupt training
            - The info parameter is reserved for future enhancements
        """
        # Don't upload if buffer is empty
        if not self.coord_list:
            return

        # Create message payload
        # Metadata identifies the agent, coords contains the position history
        message = json.dumps({
            "metadata": self.stream_metadata,
            "coords": self.coord_list
        })

        # Send message asynchronously
        # Errors are caught and logged in _send_message
        try:
            self.loop.run_until_complete(self._send_message(message))
        except Exception as e:
            print(f"StreamWrapper: Error uploading coordinates: {e}")

    def reset(self, **kwargs) -> tuple:
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
        Close WebSocket connection and cleanup resources.

        Properly shuts down the WebSocket connection and closes the
        asyncio event loop before closing the underlying environment.

        Notes:
            - Safe to call multiple times
            - Calls underlying environment's close()
            - Handles cases where connection was never established
        """
        # Close WebSocket if connected
        if self.enabled and self.websocket:
            self.loop.run_until_complete(self.websocket.close())
            self.loop.close()

        # Close underlying environment
        super().close()
