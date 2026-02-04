"""
Memory reading utilities for Pokemon Red.

This module provides documented memory addresses and clean reading functions
for extracting game state from the Pokemon Red ROM via PyBoy emulator.

The GameBoy's memory is directly accessible through PyBoy, allowing us to read
the game's internal state without relying on screen parsing. This is both faster
and more reliable than computer vision approaches.

Memory Map Reference (Pokemon Red US v1.0):
    - WRAM Bank 0: 0xC000-0xCFFF (4KB) - Temporary/scratch data
    - WRAM Bank 1: 0xD000-0xDFFF (4KB) - Persistent game state
    - Player state: 0xD300-0xD400
    - Party Pokemon: 0xD163-0xD272
    - Battle state: 0xD057-0xD100
    - Event flags: 0xD747-0xD87E (312 bytes, 2496 flags)
    - Items: 0xD31D-0xD346

Architecture Role:
    This module is the single source of truth for memory addresses. All other
    modules (env.py, rewards.py, stream_wrapper.py) import from here rather
    than defining their own addresses. This ensures consistency and makes it
    easy to update addresses if needed.

Key Concepts:
    - Little-endian: Multi-byte values store the least significant byte first.
      Example: 0x1234 is stored as [0x34, 0x12] in memory.
    - BCD (Binary-Coded Decimal): Each nibble (4 bits) represents a decimal digit.
      Used for money to avoid floating-point display issues.
    - Bitfields: Multiple boolean flags packed into a single byte.
      Badges use this: bit 0 = Boulder Badge, bit 7 = Earth Badge.

Source References:
    - pokered disassembly: https://github.com/pret/pokered
    - Data Crystal Pokemon Red RAM map: https://datacrystal.romhacking.net/wiki/Pokemon_Red/Blue:RAM_map

Note: All addresses are for the US English version of Pokemon Red (v1.0).
      Other versions (Japanese, European, Blue) may have different addresses.

Dependencies:
    - numpy: For efficient array operations on event flags
    - pyboy: GameBoy emulator (type hints only, actual import is deferred)
"""

from typing import TYPE_CHECKING

import numpy as np

# TYPE_CHECKING is False at runtime, True during static analysis
# This allows type hints without importing PyBoy at runtime (which is slow)
if TYPE_CHECKING:
    from pyboy import PyBoy


# =============================================================================
# MEMORY ADDRESS CONSTANTS - PLAYER POSITION
# =============================================================================
# These addresses track where the player is in the game world.
# The game uses a map ID system where each area has a unique identifier.

# Current Map ID
# Address: 0xD35E
# Size: 1 byte
# Range: 0-255 (though Pokemon Red uses ~248 maps)
# Source: pokered disassembly (wCurMap)
# Examples: 0 = Pallet Town, 1 = Viridian City, 40 = Indigo Plateau
ADDR_MAP_ID = 0xD35E

# Player Y Coordinate (vertical position on current map)
# Address: 0xD361
# Size: 1 byte
# Range: 0-255 (map-dependent, typically 0-44 for outdoor maps)
# Source: pokered disassembly (wYCoord)
# Note: Y increases going DOWN (screen coordinate system)
ADDR_PLAYER_Y = 0xD361

# Player X Coordinate (horizontal position on current map)
# Address: 0xD362
# Size: 1 byte
# Range: 0-255 (map-dependent, typically 0-44 for outdoor maps)
# Source: pokered disassembly (wXCoord)
# Note: X increases going RIGHT
ADDR_PLAYER_X = 0xD362


# =============================================================================
# MEMORY ADDRESS CONSTANTS - PLAYER PROGRESS
# =============================================================================
# These addresses track the player's overall progress through the game.

# Gym Badges Bitfield
# Address: 0xD356
# Size: 1 byte (8 bits = 8 badges)
# Format: Each bit represents one badge (1 = obtained, 0 = not obtained)
# Bit layout:
#   Bit 0: Boulder Badge (Brock, Pewter City)
#   Bit 1: Cascade Badge (Misty, Cerulean City)
#   Bit 2: Thunder Badge (Lt. Surge, Vermilion City)
#   Bit 3: Rainbow Badge (Erika, Celadon City)
#   Bit 4: Soul Badge (Koga, Fuchsia City)
#   Bit 5: Marsh Badge (Sabrina, Saffron City)
#   Bit 6: Volcano Badge (Blaine, Cinnabar Island)
#   Bit 7: Earth Badge (Giovanni, Viridian City)
# Source: pokered disassembly (wObtainedBadges)
ADDR_BADGES = 0xD356

# Player Money
# Address: 0xD347
# Size: 3 bytes
# Format: BCD (Binary-Coded Decimal), NOT little-endian integer
# Range: 0-999999 (max displayable in-game)
# Source: pokered disassembly (wPlayerMoney)
# Note: Each byte holds two decimal digits (0x99 = 99 decimal)
ADDR_MONEY = 0xD347


# =============================================================================
# MEMORY ADDRESS CONSTANTS - PARTY POKEMON
# =============================================================================
# The player's party can hold up to 6 Pokemon. Data is stored in arrays
# where each Pokemon occupies a fixed offset from the base address.

# Number of Pokemon in Party
# Address: 0xD163
# Size: 1 byte
# Range: 0-6 (game enforces max of 6)
# Source: pokered disassembly (wPartyCount)
ADDR_PARTY_COUNT = 0xD163

# Party Pokemon Species IDs
# Address: 0xD164
# Size: 6 bytes (1 byte per slot)
# Range: 1-190 for valid Pokemon (0 = empty, 255 = terminator)
# Source: pokered disassembly (wPartySpecies)
# Note: Species ID, not Pokedex number (they differ for some Pokemon)
ADDR_PARTY_SPECIES = 0xD164

# Party Pokemon Current HP
# Address: 0xD16C
# Size: 12 bytes (2 bytes per Pokemon, 6 Pokemon)
# Format: Little-endian 16-bit integers
# Range: 0 to max HP (up to 714 for Chansey at level 100)
# Source: pokered disassembly (wPartyMon1HP)
# Note: HP is stored in the Pokemon data structure, not as a separate array
ADDR_PARTY_HP = 0xD16C

# Party Pokemon Max HP
# Address: 0xD18D
# Size: 12 bytes (2 bytes per Pokemon, 6 Pokemon)
# Format: Little-endian 16-bit integers
# Source: pokered disassembly (wPartyMon1MaxHP)
# Note: Offset +33 bytes from current HP in the Pokemon data structure
ADDR_PARTY_MAX_HP = 0xD18D

# Party Pokemon Levels
# Address: 0xD18C
# Size: 6 bytes (1 byte per Pokemon)
# Range: 1-100
# Source: pokered disassembly (wPartyMon1Level)
# Note: Offset +32 bytes from current HP
ADDR_PARTY_LEVEL = 0xD18C


# =============================================================================
# MEMORY ADDRESS CONSTANTS - BATTLE STATE
# =============================================================================
# These addresses are only valid during battles. Outside of battle, they
# may contain stale data from the previous battle.

# Battle Active Flag
# Address: 0xD057
# Size: 1 byte
# Values: 0 = not in battle, non-zero = in battle
# Source: pokered disassembly (wIsInBattle)
# Note: Check this before reading other battle addresses
ADDR_IN_BATTLE = 0xD057

# Battle Type
# Address: 0xD058
# Size: 1 byte
# Values: 0 = wild battle, 1 = trainer battle, 2 = scripted battle
# Source: pokered disassembly (wBattleType)
ADDR_BATTLE_TYPE = 0xD058

# Player's Active Pokemon Current HP (in battle)
# Address: 0xD015
# Size: 2 bytes (little-endian)
# Source: pokered disassembly (wBattleMonHP)
# Note: This is the battle copy, not the party data
ADDR_PLAYER_HP = 0xD015

# Player's Active Pokemon Max HP (in battle)
# Address: 0xD017
# Size: 2 bytes (little-endian)
# Source: pokered disassembly (wBattleMonMaxHP)
ADDR_PLAYER_MAX_HP = 0xD017

# Enemy Pokemon Current HP
# Address: 0xCFE6
# Size: 2 bytes (little-endian)
# Source: pokered disassembly (wEnemyMonHP)
# Note: For wild Pokemon and trainer Pokemon alike
ADDR_ENEMY_HP = 0xCFE6

# Enemy Pokemon Max HP
# Address: 0xCFE8
# Size: 2 bytes (little-endian)
# Source: pokered disassembly (wEnemyMonMaxHP)
ADDR_ENEMY_MAX_HP = 0xCFE8


# =============================================================================
# MEMORY ADDRESS CONSTANTS - EVENT FLAGS
# =============================================================================
# Event flags are a large bitfield tracking game progress. Each flag represents
# a specific event: items picked up, trainers defeated, story triggers, etc.

# Event Flags Start Address
# Address: 0xD747
# Size: 312 bytes (2496 individual flags)
# Format: Bitfield (8 flags per byte)
# Source: pokered disassembly (wEventFlags)
# Note: Some flags are volatile (reset on map change), others are permanent
ADDR_EVENT_FLAGS = 0xD747

# Number of bytes containing event flags
# 312 bytes * 8 bits = 2496 actual flags
# We round up to 2560 for observation space (multiple of 256 for efficiency)
EVENT_FLAGS_SIZE = 312


# =============================================================================
# MEMORY ADDRESS CONSTANTS - UI STATE
# =============================================================================
# These addresses help detect when the game is waiting for input or showing
# menus/text, which affects how actions should be interpreted.

# Menu Open Flag
# Address: 0xCC26
# Size: 1 byte
# Values: 0 = no menu, non-zero = menu is open
# Source: pokered disassembly (wMenuJoypadPollCount or similar)
# Note: Includes start menu, shop menus, PC menus, etc.
ADDR_MENU_OPEN = 0xCC26

# Text Box Displayed Flag
# Address: 0xCC3C
# Size: 1 byte
# Values: 0 = no text box, non-zero = text box visible
# Source: pokered disassembly (wDoNotWaitForButtonPressAfterDisplayingText)
# Note: Indicates NPC dialogue, sign text, item descriptions, etc.
ADDR_TEXT_BOX = 0xCC3C


# =============================================================================
# LOW-LEVEL MEMORY READING FUNCTIONS
# =============================================================================
# These functions provide the basic building blocks for reading different
# data types from GameBoy memory.


def read_byte(pyboy: "PyBoy", addr: int) -> int:
    """
    Read a single byte from GameBoy memory.

    This is the most basic memory read operation. All other read functions
    ultimately use this to access memory.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        addr: Memory address to read from (0x0000-0xFFFF).

    Returns:
        Integer value of the byte (0-255).

    Example:
        >>> badges = read_byte(pyboy, ADDR_BADGES)
        >>> print(f"Badge bitfield: {badges:08b}")
        Badge bitfield: 00000011  # Boulder and Cascade badges

    Notes:
        - PyBoy provides direct memory access via the memory attribute
        - Addresses outside valid ranges may return garbage or cause errors
        - This function does not validate the address
    """
    return pyboy.memory[addr]


def read_word(pyboy: "PyBoy", addr: int) -> int:
    """
    Read a 16-bit word (2 bytes) from GameBoy memory in little-endian format.

    The GameBoy uses little-endian byte order, meaning the least significant
    byte is stored at the lower address. For example, the value 0x1234 is
    stored as [0x34, 0x12] in consecutive memory locations.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        addr: Memory address of the low byte (the word spans addr and addr+1).

    Returns:
        Integer value of the 16-bit word (0-65535).

    Example:
        >>> hp = read_word(pyboy, ADDR_PARTY_HP)
        >>> print(f"First Pokemon HP: {hp}")
        First Pokemon HP: 45

    Notes:
        - Used for HP values, stats, and other 16-bit quantities
        - The GameBoy CPU (Z80-like) is natively little-endian
        - addr+1 must also be a valid memory address
    """
    # Read the low byte from the base address
    lo = pyboy.memory[addr]
    # Read the high byte from the next address
    hi = pyboy.memory[addr + 1]
    # Combine: shift high byte left 8 bits, then OR with low byte
    return (hi << 8) | lo


def read_bcd(pyboy: "PyBoy", addr: int, n_bytes: int = 3) -> int:
    """
    Read a BCD (Binary-Coded Decimal) encoded number from memory.

    BCD stores decimal digits in nibbles (4 bits each). Each byte holds two
    decimal digits: the high nibble (bits 4-7) is the tens digit, and the
    low nibble (bits 0-3) is the ones digit.

    Pokemon Red uses BCD for the player's money to avoid floating-point
    display issues and make it easy to show exact decimal values.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        addr: Starting memory address of the BCD number.
        n_bytes: Number of bytes to read (default 3 for money = 6 digits).

    Returns:
        Integer value of the decoded BCD number.

    Example:
        >>> money = read_bcd(pyboy, ADDR_MONEY, 3)
        >>> print(f"Player has ${money}")
        Player has $3000

        Memory layout for $3000:
        Byte 0: 0x00 (00 in decimal)
        Byte 1: 0x30 (30 in decimal)
        Byte 2: 0x00 (00 in decimal)
        Result: 00 * 10000 + 30 * 100 + 00 = 3000

    Notes:
        - Each byte contributes two decimal digits (0-99)
        - Invalid BCD (nibble > 9) may produce unexpected results
        - Money in Pokemon Red is capped at $999,999
    """
    result = 0
    for i in range(n_bytes):
        byte = pyboy.memory[addr + i]
        # Extract high nibble (tens digit) and low nibble (ones digit)
        # High nibble: shift right 4 bits, multiply by 10
        # Low nibble: mask with 0x0F to get lower 4 bits
        # Combined: high_nibble * 10 + low_nibble = two decimal digits
        result = result * 100 + ((byte >> 4) * 10) + (byte & 0x0F)
    return result


# =============================================================================
# GAME STATE READING FUNCTIONS - POSITION
# =============================================================================
# Functions for reading the player's current location in the game world.


def get_position(pyboy: "PyBoy") -> tuple[int, int, int]:
    """
    Get the player's current position in the game world.

    Returns the map ID and local coordinates within that map. The coordinate
    system has (0,0) at the top-left corner of the map, with X increasing
    rightward and Y increasing downward.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Tuple of (map_id, x, y) where:
        - map_id: Unique identifier for the current map (0-255)
        - x: Horizontal position within the map
        - y: Vertical position within the map

    Example:
        >>> map_id, x, y = get_position(pyboy)
        >>> from kantorl.global_map import get_map_name
        >>> print(f"Player is at ({x}, {y}) in {get_map_name(map_id)}")
        Player is at (5, 6) in Pallet Town

    Notes:
        - Coordinates are local to the current map
        - Use global_map.local_to_global() for world-space coordinates
        - Indoor maps have separate coordinate systems from outdoor maps
    """
    map_id = read_byte(pyboy, ADDR_MAP_ID)
    x = read_byte(pyboy, ADDR_PLAYER_X)
    y = read_byte(pyboy, ADDR_PLAYER_Y)
    return map_id, x, y


# =============================================================================
# GAME STATE READING FUNCTIONS - PROGRESS
# =============================================================================
# Functions for reading the player's progress through the game.


def get_badges(pyboy: "PyBoy") -> int:
    """
    Get the total number of gym badges the player has earned.

    Counts the number of set bits in the badge bitfield, giving a simple
    progress metric from 0 (no badges) to 8 (all badges).

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Number of badges earned (0-8).

    Example:
        >>> badges = get_badges(pyboy)
        >>> print(f"Player has {badges}/8 badges")
        Player has 3/8 badges

    Notes:
        - This counts total badges, not which specific badges
        - Use get_badge_flags() for individual badge status
        - Badges are a key milestone reward in training
    """
    badge_bits = read_byte(pyboy, ADDR_BADGES)
    # Count the number of 1 bits in the badge bitfield
    # bin() converts to binary string like "0b11", count("1") counts the 1s
    return bin(badge_bits).count("1")


def get_badge_flags(pyboy: "PyBoy") -> np.ndarray:
    """
    Get individual badge flags as an 8-element binary array.

    Each element corresponds to one gym badge in order: Boulder, Cascade,
    Thunder, Rainbow, Soul, Marsh, Volcano, Earth.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        NumPy array of shape (8,) with dtype int8.
        Each element is 0 (not obtained) or 1 (obtained).

    Example:
        >>> flags = get_badge_flags(pyboy)
        >>> badge_names = ["Boulder", "Cascade", "Thunder", "Rainbow",
        ...               "Soul", "Marsh", "Volcano", "Earth"]
        >>> for name, flag in zip(badge_names, flags):
        ...     status = "✓" if flag else "✗"
        ...     print(f"{name}: {status}")

    Notes:
        - Array index 0 = Boulder Badge (Brock), index 7 = Earth Badge (Giovanni)
        - Used in observation space to give the agent badge information
        - int8 dtype for memory efficiency in large batch operations
    """
    badge_bits = read_byte(pyboy, ADDR_BADGES)
    # Extract each bit into a separate array element
    # (badge_bits >> i) & 1 extracts bit i (0 = LSB, 7 = MSB)
    return np.array([(badge_bits >> i) & 1 for i in range(8)], dtype=np.int8)


def get_money(pyboy: "PyBoy") -> int:
    """
    Get the player's current money amount.

    Money in Pokemon Red is stored as a 3-byte BCD number, allowing values
    from 0 to 999,999.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Player's money as an integer (0-999999).

    Example:
        >>> money = get_money(pyboy)
        >>> print(f"Player has ${money:,}")
        Player has $12,500

    Notes:
        - Money is used to buy items, but not directly useful for RL rewards
        - The BCD format means memory reads require special decoding
        - Max money is $999,999 in the game
    """
    return read_bcd(pyboy, ADDR_MONEY, 3)


# =============================================================================
# GAME STATE READING FUNCTIONS - PARTY POKEMON
# =============================================================================
# Functions for reading information about the player's Pokemon team.


def get_party_count(pyboy: "PyBoy") -> int:
    """
    Get the number of Pokemon currently in the player's party.

    The party can hold 0-6 Pokemon. At game start with a starter, this is 1.
    The value is clamped to 6 to handle any memory corruption.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Number of Pokemon in party (0-6).

    Example:
        >>> count = get_party_count(pyboy)
        >>> print(f"Party has {count} Pokemon")
        Party has 3 Pokemon

    Notes:
        - Returns 0 before receiving starter Pokemon
        - Clamped to 6 as a safety measure against invalid memory states
        - Used to determine how many party slots to read
    """
    # Clamp to 6 to prevent reading invalid data if memory is corrupted
    return min(read_byte(pyboy, ADDR_PARTY_COUNT), 6)


def get_party_hp(pyboy: "PyBoy") -> list[tuple[int, int]]:
    """
    Get HP values for all Pokemon in the party.

    Returns a list of (current_hp, max_hp) tuples, one per Pokemon in the
    party. The list length matches the party count.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        List of tuples [(current_hp, max_hp), ...] for each party Pokemon.
        Empty list if party is empty.

    Example:
        >>> hp_list = get_party_hp(pyboy)
        >>> for i, (current, maximum) in enumerate(hp_list):
        ...     print(f"Pokemon {i+1}: {current}/{maximum} HP")
        Pokemon 1: 45/45 HP
        Pokemon 2: 30/52 HP
        Pokemon 3: 0/38 HP  # Fainted

    Notes:
        - HP values are 16-bit (0-65535), though real max is ~714
        - Fainted Pokemon have current_hp = 0
        - Used to calculate party health fraction for observations
    """
    count = get_party_count(pyboy)
    result = []
    for i in range(count):
        # HP values are 16-bit, stored every 2 bytes
        current = read_word(pyboy, ADDR_PARTY_HP + i * 2)
        max_hp = read_word(pyboy, ADDR_PARTY_MAX_HP + i * 2)
        result.append((current, max_hp))
    return result


def get_party_levels(pyboy: "PyBoy") -> list[int]:
    """
    Get the levels of all Pokemon in the party.

    Returns a list of level values, one per Pokemon in the party.
    Levels range from 1 to 100.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        List of integers representing each Pokemon's level (1-100).
        Empty list if party is empty.

    Example:
        >>> levels = get_party_levels(pyboy)
        >>> print(f"Party levels: {levels}")
        Party levels: [15, 12, 10]
        >>> print(f"Average level: {sum(levels)/len(levels):.1f}")
        Average level: 12.3

    Notes:
        - Levels are single bytes (1-100 in normal gameplay)
        - Used for Fourier encoding in observation space
        - Level ups are an important progress indicator
    """
    count = get_party_count(pyboy)
    # Levels are stored as single bytes, one per Pokemon
    return [read_byte(pyboy, ADDR_PARTY_LEVEL + i) for i in range(count)]


def get_total_party_hp(pyboy: "PyBoy") -> tuple[int, int]:
    """
    Get the total HP across all party Pokemon.

    Sums the current and max HP of all party members, providing a single
    metric for overall party health status.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Tuple of (total_current_hp, total_max_hp).
        Returns (0, 0) if party is empty.

    Example:
        >>> current, maximum = get_total_party_hp(pyboy)
        >>> health_pct = current / maximum * 100 if maximum > 0 else 0
        >>> print(f"Party health: {current}/{maximum} ({health_pct:.0f}%)")
        Party health: 150/200 (75%)

    Notes:
        - Used to calculate party health fraction for observations
        - A ratio of 0 means all Pokemon are fainted (game over soon)
        - Healing rewards are based on changes to this value
    """
    hp_list = get_party_hp(pyboy)
    if not hp_list:
        return 0, 0
    # Sum current HP and max HP separately across all party Pokemon
    current = sum(hp for hp, _ in hp_list)
    max_hp = sum(max_hp for _, max_hp in hp_list)
    return current, max_hp


# =============================================================================
# GAME STATE READING FUNCTIONS - BATTLE
# =============================================================================
# Functions for reading battle-specific state. Only valid during battles.


def is_in_battle(pyboy: "PyBoy") -> bool:
    """
    Check if the player is currently in a battle.

    This should be checked before reading other battle-related memory,
    as those addresses contain stale data outside of battles.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        True if in battle, False otherwise.

    Example:
        >>> if is_in_battle(pyboy):
        ...     player_hp, enemy_hp = get_battle_hp(pyboy)
        ...     print(f"Battle! Player: {player_hp:.0%}, Enemy: {enemy_hp:.0%}")
        ... else:
        ...     print("Exploring the world...")

    Notes:
        - Battle flag is non-zero during wild and trainer battles
        - The flag may briefly be set during battle transitions
        - Used in env.py to add battle state to observations
    """
    return read_byte(pyboy, ADDR_IN_BATTLE) != 0


def get_battle_hp(pyboy: "PyBoy") -> tuple[float, float]:
    """
    Get HP ratios for both Pokemon in battle.

    Returns the current HP as a fraction of max HP for both the player's
    active Pokemon and the enemy Pokemon. This provides normalized values
    suitable for use in observations or rewards.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Tuple of (player_ratio, enemy_ratio) where each is a float 0.0-1.0.
        Returns (1.0, 0.0) if not in battle.

    Example:
        >>> player_hp, enemy_hp = get_battle_hp(pyboy)
        >>> print(f"Player: {player_hp:.0%} HP")
        >>> print(f"Enemy: {enemy_hp:.0%} HP")
        Player: 75% HP
        Enemy: 50% HP

    Notes:
        - Returns (1.0, 0.0) outside battle for safe default values
        - Division by max(hp, 1) prevents division by zero
        - Ratios are clamped naturally to [0, 1] by HP constraints
    """
    if not is_in_battle(pyboy):
        # Default values when not in battle
        return 1.0, 0.0

    # Read raw HP values
    player_hp = read_word(pyboy, ADDR_PLAYER_HP)
    player_max = read_word(pyboy, ADDR_PLAYER_MAX_HP)
    enemy_hp = read_word(pyboy, ADDR_ENEMY_HP)
    enemy_max = read_word(pyboy, ADDR_ENEMY_MAX_HP)

    # Calculate ratios, avoiding division by zero with max(..., 1)
    player_ratio = player_hp / max(player_max, 1)
    enemy_ratio = enemy_hp / max(enemy_max, 1)

    return player_ratio, enemy_ratio


# =============================================================================
# GAME STATE READING FUNCTIONS - EVENT FLAGS
# =============================================================================
# Functions for reading the event flag bitfield that tracks game progress.


def get_event_flags(pyboy: "PyBoy") -> np.ndarray:
    """
    Get all event flags as a binary numpy array.

    Event flags track nearly everything in the game: items picked up,
    trainers defeated, story events completed, Pokemon caught, etc.
    This provides a rich signal for measuring game progress.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        NumPy array of shape (2560,) with dtype int8.
        Each element is 0 (flag not set) or 1 (flag set).

    Example:
        >>> flags = get_event_flags(pyboy)
        >>> set_flags = np.sum(flags)
        >>> print(f"{set_flags} event flags are set")
        142 event flags are set

    Notes:
        - 312 bytes * 8 bits = 2496 actual flags, padded to 2560
        - Padding to 2560 (multiple of 256) for observation space alignment
        - Used directly in the observation space for the agent
        - Many flags are permanently set, some reset on map change
    """
    # Pre-allocate array for all flags
    flags = np.zeros(EVENT_FLAGS_SIZE * 8, dtype=np.int8)

    # Extract each bit from each byte
    for i in range(EVENT_FLAGS_SIZE):
        byte = read_byte(pyboy, ADDR_EVENT_FLAGS + i)
        # Unpack 8 bits from this byte into the array
        for bit in range(8):
            # (byte >> bit) & 1 extracts bit number 'bit'
            flags[i * 8 + bit] = (byte >> bit) & 1

    # Pad to 2560 for consistent observation space size
    # np.pad adds zeros to the end to reach target length
    return np.pad(flags, (0, 2560 - len(flags)))


def count_event_flags(pyboy: "PyBoy") -> int:
    """
    Count the total number of set event flags.

    Provides a single integer metric for game progress without needing
    to process the full flag array.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Number of event flags that are set (0 to ~2496).

    Example:
        >>> events = count_event_flags(pyboy)
        >>> print(f"Progress: {events} events completed")
        Progress: 142 events completed

    Notes:
        - More efficient than get_event_flags() when only count is needed
        - Used in reward calculation to detect new events
        - Typical endgame value is ~1000+ flags
    """
    count = 0
    for i in range(EVENT_FLAGS_SIZE):
        byte = read_byte(pyboy, ADDR_EVENT_FLAGS + i)
        # Count 1 bits in this byte using bin().count("1")
        count += bin(byte).count("1")
    return count


# =============================================================================
# GAME STATE READING FUNCTIONS - UI STATE
# =============================================================================
# Functions for detecting UI states that affect gameplay.


def is_menu_open(pyboy: "PyBoy") -> bool:
    """
    Check if a menu is currently open.

    Menus include the start menu, shop menus, PC menus, battle menus, etc.
    When a menu is open, only certain actions are valid.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        True if any menu is open, False otherwise.

    Example:
        >>> if is_menu_open(pyboy):
        ...     print("Menu is open - use A/B to navigate")
        ... else:
        ...     print("World view - use arrows to move")

    Notes:
        - Not currently used in observations but available for future use
        - Could be used to modify action masking or rewards
        - Menu state affects which buttons do what
    """
    return read_byte(pyboy, ADDR_MENU_OPEN) != 0


def is_text_displayed(pyboy: "PyBoy") -> bool:
    """
    Check if a text box is currently displayed.

    Text boxes appear for NPC dialogue, sign text, item descriptions,
    battle messages, and other text content. The game waits for button
    press to advance text.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        True if text box is visible, False otherwise.

    Example:
        >>> if is_text_displayed(pyboy):
        ...     print("Text displayed - press A or B to continue")

    Notes:
        - Not currently used in observations but available for future use
        - Could be used to encourage the agent to press A/B to advance
        - Text boxes pause normal gameplay
    """
    return read_byte(pyboy, ADDR_TEXT_BOX) != 0
