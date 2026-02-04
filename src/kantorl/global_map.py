"""
Global map coordinate conversion for Pokemon Red.

This module converts local map coordinates (within a single area like "Pallet Town")
to global coordinates on a unified Kanto region map. This enables:
- Visualization of the agent's exploration on a single cohesive map
- Tracking unique global coordinates for exploration rewards
- Heatmap generation showing where the agent spends time

Architecture Role:
    The global_map module provides coordinate transformation utilities:
    - local_to_global(): Convert (map_id, x, y) → (global_x, global_y)
    - get_map_name(): Get human-readable name for a map ID
    - is_outdoor_map(): Check if a map is part of the overworld

    This is used by the environment for exploration tracking and by
    visualization tools to render agent progress on a Kanto map.

Coordinate Systems:
    Pokemon Red uses two coordinate systems:

    1. Local Coordinates:
       - Each map (Pallet Town, Route 1, etc.) has its own coordinate grid
       - Origin (0,0) is the top-left corner of each map
       - X increases rightward, Y increases downward
       - Indoor maps have small grids (e.g., 4x4 for houses)
       - Outdoor maps can be larger (e.g., 20x18 for Pallet Town)

    2. Global Coordinates:
       - A single large grid covering all of Kanto
       - All outdoor maps are positioned on this grid
       - Indoor maps are typically not part of the global map (return 0,0)
       - The global map is 444 x 436 tiles

    Conversion uses map_data.json which stores the base (x, y) position
    of each map on the global grid.

Map Data Format:
    The map_data.json file contains entries like:
    {
        "0": {"name": "Pallet Town", "x": 24, "y": 360, "outdoor": true},
        "12": {"name": "Route 1", "x": 24, "y": 324, "outdoor": true},
        ...
    }

    Where:
    - Key is the map_id as a string
    - "name" is the human-readable location name
    - "x", "y" are the base coordinates on the global map
    - "outdoor" indicates if this is an overworld map

Usage:
    >>> from kantorl.global_map import local_to_global, get_map_name
    >>> # Player is at (5, 3) in Pallet Town (map_id=0)
    >>> global_x, global_y = local_to_global(0, 5, 3)
    >>> print(f"Global position: ({global_x}, {global_y})")
    Global position: (29, 363)

    >>> print(get_map_name(0))
    Pallet Town

Dependencies:
    - json: For loading map_data.json
    - pathlib: For locating the data file

Data Files:
    - data/map_data.json: Contains map positions and metadata
"""

import json
from pathlib import Path


# =============================================================================
# GLOBAL MAP CONSTANTS
# =============================================================================

# Global map dimensions in tiles
# These represent the size of the unified Kanto region map
# All outdoor map coordinates fall within these bounds

# Width of the global Kanto map in tiles (horizontal extent)
# Covers from the western edge of Kanto to the eastern edge
GLOBAL_MAP_WIDTH = 444

# Height of the global Kanto map in tiles (vertical extent)
# Covers from the northern edge (Indigo Plateau) to southern edge (Pallet Town area)
GLOBAL_MAP_HEIGHT = 436


# =============================================================================
# MAP DATA LOADING
# =============================================================================

# Module-level cache for map data
# Loaded lazily on first access to avoid startup cost
# Contains: {"map_id": {"name": str, "x": int, "y": int, "outdoor": bool}, ...}
_MAP_DATA: dict | None = None


def _load_map_data() -> dict:
    """
    Load map coordinate data from the JSON file.

    This function implements lazy loading with caching. The map data is only
    loaded once on first access, then cached in the module-level _MAP_DATA
    variable for subsequent calls.

    The data file (map_data.json) contains the position of each map on the
    global Kanto grid, along with metadata like the map name and whether
    it's an outdoor map.

    Returns:
        Dictionary mapping map_id strings to map info dictionaries.
        Each map info contains:
        - "name": Human-readable location name
        - "x": Base X coordinate on global map
        - "y": Base Y coordinate on global map
        - "outdoor": Boolean indicating if this is an overworld map

        Returns empty dict if the data file is not found.

    Example:
        >>> data = _load_map_data()
        >>> data["0"]
        {"name": "Pallet Town", "x": 24, "y": 360, "outdoor": True}

    Notes:
        - Uses global _MAP_DATA for caching (not ideal but simple)
        - Data file path is relative to this module's location
        - Missing data file results in empty dict (graceful degradation)
    """
    global _MAP_DATA

    # Return cached data if already loaded
    if _MAP_DATA is None:
        # Construct path to data file relative to this module
        # __file__ is the path to this Python file (global_map.py)
        # Parent directory contains the data/ subdirectory
        data_path = Path(__file__).parent / "data" / "map_data.json"

        if data_path.exists():
            # Load and parse the JSON file
            with open(data_path) as f:
                _MAP_DATA = json.load(f)
        else:
            # Fallback: empty map data (all maps will return (0, 0))
            # This allows the code to run even without the data file
            _MAP_DATA = {}

    return _MAP_DATA


# =============================================================================
# COORDINATE CONVERSION
# =============================================================================


def local_to_global(map_id: int, x: int, y: int) -> tuple[int, int]:
    """
    Convert local map coordinates to global Kanto region position.

    Takes a position within a specific map (e.g., Pallet Town) and converts
    it to coordinates on the unified global Kanto map. This is essential for:
    - Tracking exploration across map boundaries
    - Visualizing the agent's journey on a single map
    - Computing distances between locations

    Conversion Formula:
        global_x = map_base_x + local_x
        global_y = map_base_y + local_y

    Where (map_base_x, map_base_y) is the position of the map's top-left
    corner on the global grid, stored in map_data.json.

    Args:
        map_id: Current map ID from game memory (0-255).
               Maps are numbered areas like Pallet Town (0), Route 1 (12),
               Viridian City (1), etc.
        x: Local X coordinate within the map (0 = leftmost column).
        y: Local Y coordinate within the map (0 = topmost row).

    Returns:
        Tuple of (global_x, global_y) position on the Kanto region map.
        Returns (0, 0) for:
        - Unknown map IDs not in the data file
        - Indoor maps that aren't part of the global overworld

    Example:
        >>> # Player at position (5, 3) in Pallet Town (map_id=0)
        >>> # Pallet Town's base position is (24, 360)
        >>> global_x, global_y = local_to_global(0, 5, 3)
        >>> print(global_x, global_y)
        29 363

        >>> # Unknown map returns (0, 0)
        >>> local_to_global(255, 5, 5)
        (0, 0)

    Notes:
        - Indoor maps typically return (0, 0) since they're not on the overworld
        - The return value can be used directly as a key for exploration tracking
        - Global coordinates may exceed GLOBAL_MAP_WIDTH/HEIGHT for edge maps
    """
    # Load map data (cached after first call)
    map_data = _load_map_data()

    # Map IDs are stored as strings in the JSON file
    map_key = str(map_id)

    # Return (0, 0) for unknown maps
    # This handles both truly unknown maps and indoor maps
    if map_key not in map_data:
        return 0, 0

    # Get the map's base position on the global grid
    info = map_data[map_key]
    base_x = info.get("x", 0)  # Default to 0 if missing
    base_y = info.get("y", 0)  # Default to 0 if missing

    # Add local offset to base position
    return base_x + x, base_y + y


# =============================================================================
# MAP METADATA
# =============================================================================


def get_map_name(map_id: int) -> str:
    """
    Get the human-readable name for a map ID.

    Converts numeric map IDs to friendly location names for logging,
    debugging, and user-facing output.

    Args:
        map_id: Map ID from game memory (0-255).

    Returns:
        Human-readable location name, e.g., "Pallet Town", "Route 1".
        Returns "Unknown (ID)" for map IDs not in the data file.
        Returns "Map ID" if the name field is missing in the data.

    Example:
        >>> get_map_name(0)
        'Pallet Town'

        >>> get_map_name(12)
        'Route 1'

        >>> get_map_name(255)
        'Unknown (255)'

    Notes:
        - Useful for logging agent location during training/evaluation
        - Names come from the Pokemon Red disassembly/documentation
    """
    # Load map data (cached after first call)
    map_data = _load_map_data()

    # Map IDs are stored as strings in the JSON file
    map_key = str(map_id)

    # Return descriptive string for unknown maps
    if map_key not in map_data:
        return f"Unknown ({map_id})"

    # Get name from data, with fallback
    return map_data[map_key].get("name", f"Map {map_id}")


def is_outdoor_map(map_id: int) -> bool:
    """
    Check if a map ID corresponds to an outdoor/overworld map.

    Outdoor maps are part of the main Kanto region overworld and have
    valid global coordinates. Indoor maps (houses, caves, buildings)
    are separate instances not connected to the global map.

    This distinction is important for:
    - Exploration tracking (only count outdoor exploration)
    - Map visualization (indoor maps aren't on the global map)
    - Coordinate conversion (indoor maps return (0, 0))

    Args:
        map_id: Map ID from game memory (0-255).

    Returns:
        True if the map is an outdoor/overworld map.
        False for indoor maps or unknown map IDs.

    Example:
        >>> is_outdoor_map(0)  # Pallet Town
        True

        >>> is_outdoor_map(12)  # Route 1
        True

        >>> is_outdoor_map(37)  # Red's House 1F (indoor)
        False

        >>> is_outdoor_map(255)  # Unknown
        False

    Notes:
        - Caves and dungeons may be marked as outdoor=False even though
          they're "outside" buildings
        - The outdoor flag affects whether global coordinates are meaningful
    """
    # Load map data (cached after first call)
    map_data = _load_map_data()

    # Map IDs are stored as strings in the JSON file
    map_key = str(map_id)

    # Unknown maps are treated as not outdoor
    if map_key not in map_data:
        return False

    # Return the outdoor flag, defaulting to False
    return map_data[map_key].get("outdoor", False)
