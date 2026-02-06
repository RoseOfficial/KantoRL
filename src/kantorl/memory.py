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
        - Uses vectorized np.unpackbits for ~10x speedup vs pure Python loop
    """
    # Read all 312 flag bytes in one slice (avoids 312 individual read_byte calls)
    # pyboy.memory[...] returns a list of ints, so we use np.array (not frombuffer)
    raw = np.array(
        pyboy.memory[ADDR_EVENT_FLAGS : ADDR_EVENT_FLAGS + EVENT_FLAGS_SIZE],
        dtype=np.uint8,
    )

    # Unpack all bits at once: bitorder='little' matches the original (byte >> bit) & 1
    # extraction order where bit 0 is the LSB
    flags = np.unpackbits(raw, bitorder="little").astype(np.int8)

    # Pad from 2496 to 2560 for consistent observation space size
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
        - Uses vectorized np.unpackbits for efficient bit counting
        - Used in reward calculation to detect new events
        - Typical endgame value is ~1000+ flags
    """
    # Read all 312 flag bytes in one slice and count set bits via unpackbits
    # pyboy.memory[...] returns a list of ints, so we use np.array (not frombuffer)
    raw = np.array(
        pyboy.memory[ADDR_EVENT_FLAGS : ADDR_EVENT_FLAGS + EVENT_FLAGS_SIZE],
        dtype=np.uint8,
    )
    return int(np.sum(np.unpackbits(raw)))


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


# =============================================================================
# MEMORY ADDRESS CONSTANTS - BAG ITEMS
# =============================================================================
# The player's bag stores items as sequential (item_id, quantity) pairs.
# The bag can hold up to 20 different item types.

# Number of distinct items in the bag
# Address: 0xD31D
# Size: 1 byte
# Range: 0-20
# Source: pokered disassembly (wNumBagItems)
ADDR_BAG_COUNT = 0xD31D

# Start of bag item data (pairs of item_id, quantity)
# Address: 0xD31E
# Size: 2 bytes per item (item_id + quantity), up to 20 items
# Format: [item_id_0, qty_0, item_id_1, qty_1, ..., 0xFF terminator]
# Source: pokered disassembly (wBagItems)
ADDR_BAG_ITEMS = 0xD31E


# =============================================================================
# MEMORY ADDRESS CONSTANTS - PARTY POKEMON MOVES
# =============================================================================
# Each party Pokemon stores 4 move IDs and 4 PP values.
# The party data structure uses a 44-byte stride per Pokemon.

# Party Pokemon Move IDs (first Pokemon)
# Address: 0xD173
# Size: 4 bytes (1 byte per move slot)
# Range: 0-165 for valid moves (0 = empty slot)
# Source: pokered disassembly (wPartyMon1Moves)
# Note: Subsequent Pokemon at +44 byte intervals
ADDR_PARTY_MOVES = 0xD173

# Party Pokemon Move PP (first Pokemon)
# Address: 0xD188
# Size: 4 bytes (1 byte per move slot)
# Source: pokered disassembly (wPartyMon1PP)
# Note: Subsequent Pokemon at +44 byte intervals
ADDR_PARTY_PP = 0xD188

# Stride between Pokemon in the party data structure
# Each Pokemon occupies 44 bytes of data
# Source: pokered disassembly (PARTYMON_STRUCT_LENGTH)
PARTY_MON_STRIDE = 44


# =============================================================================
# MEMORY ADDRESS CONSTANTS - FIELD MOVE DETECTION
# =============================================================================
# Addresses used to detect when field moves (Cut, Surf, Strength) can be used.

# Tile ID that the player is currently facing
# Address: 0xCFC6
# Size: 1 byte
# Source: pokered disassembly (wTileInFrontOfPlayer)
# Note: Contains the tile type the player would step onto / interact with
ADDR_TILE_IN_FRONT = 0xCFC6

# Walk/Bike/Surf state
# Address: 0xD700
# Size: 1 byte
# Values: 0 = walking, 1 = biking, 2 = surfing
# Source: pokered disassembly (wWalkBikeSurfState)
ADDR_WALK_BIKE_SURF = 0xD700


# =============================================================================
# HM ITEM AND MOVE CONSTANTS
# =============================================================================
# Item IDs for Hidden Machines found in the bag, and their corresponding
# move IDs when taught to a Pokemon.

# HM Item IDs (found in bag)
# Source: pokered disassembly (constants/item_constants.asm)
HM_CUT_ITEM = 0xC4      # HM01 Cut (item ID 196)
HM_SURF_ITEM = 0xC6     # HM03 Surf (item ID 198)
HM_STRENGTH_ITEM = 0xC7  # HM04 Strength (item ID 199)

# Corresponding Move IDs (written to Pokemon move slots)
# Source: pokered disassembly (constants/move_constants.asm)
MOVE_CUT = 0x0F          # Move ID 15 - Cut
MOVE_SURF = 0x39         # Move ID 57 - Surf
MOVE_STRENGTH = 0x46     # Move ID 70 - Strength

# PP values for HM moves (base PP, no PP Ups)
# Source: pokered disassembly (data/moves/moves.asm)
PP_CUT = 30
PP_SURF = 15
PP_STRENGTH = 15

# Cuttable tree tile IDs (overworld tiles that can be cut)
# Source: pokered disassembly (engine/overworld/cut.asm)
# These are the tile IDs that represent small trees blocking the path
CUTTABLE_TREE_TILES = frozenset({0x3D, 0x50})

# Water tile IDs (tiles that can be surfed on)
# Source: pokered disassembly (engine/overworld/player_movement.asm)
WATER_TILES = frozenset({0x14, 0x32, 0x48})


# =============================================================================
# GAME STATE READING FUNCTIONS - BAG AND ITEMS
# =============================================================================
# Functions for reading the player's inventory.


def get_bag_items(pyboy: "PyBoy") -> list[tuple[int, int]]:
    """
    Get all items in the player's bag.

    Items are stored as sequential (item_id, quantity) pairs in memory,
    terminated by 0xFF. The count at ADDR_BAG_COUNT tells us how many
    pairs to read.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        List of (item_id, quantity) tuples for each item in the bag.
        Empty list if bag is empty.

    Example:
        >>> items = get_bag_items(pyboy)
        >>> for item_id, qty in items:
        ...     print(f"Item 0x{item_id:02X}: x{qty}")
        Item 0xC4: x1  # HM01 Cut

    Notes:
        - Item count is clamped to 20 (max bag capacity)
        - Each item occupies 2 bytes: item_id at offset 0, quantity at offset 1
        - HM items always have quantity 1 and cannot be discarded
    """
    count = min(read_byte(pyboy, ADDR_BAG_COUNT), 20)
    items = []
    for i in range(count):
        item_id = read_byte(pyboy, ADDR_BAG_ITEMS + i * 2)
        quantity = read_byte(pyboy, ADDR_BAG_ITEMS + i * 2 + 1)
        items.append((item_id, quantity))
    return items


def has_item(pyboy: "PyBoy", item_id: int) -> bool:
    """
    Check if the player has a specific item in their bag.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        item_id: The item ID to search for (e.g., HM_CUT_ITEM = 0xC4).

    Returns:
        True if the item is in the bag, False otherwise.

    Example:
        >>> if has_item(pyboy, HM_CUT_ITEM):
        ...     print("Player has HM01 Cut!")
    """
    return any(iid == item_id for iid, _ in get_bag_items(pyboy))


# =============================================================================
# GAME STATE READING FUNCTIONS - PARTY MOVES
# =============================================================================
# Functions for reading and writing Pokemon move data.


def get_party_moves(pyboy: "PyBoy", slot: int) -> list[int]:
    """
    Get the 4 move IDs for a party Pokemon.

    Each Pokemon has 4 move slots. Empty slots contain 0.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        slot: Party slot index (0-5, where 0 is the first Pokemon).

    Returns:
        List of 4 move IDs (0 = empty slot).

    Example:
        >>> moves = get_party_moves(pyboy, 0)
        >>> print(f"First Pokemon moves: {moves}")
        First Pokemon moves: [33, 45, 0, 0]  # Tackle, Growl, empty, empty

    Notes:
        - Slot must be 0-5 (not bounds-checked for performance)
        - Move ID 0 means the slot is empty
        - Moves are stored at ADDR_PARTY_MOVES + slot * PARTY_MON_STRIDE
    """
    base = ADDR_PARTY_MOVES + slot * PARTY_MON_STRIDE
    return [read_byte(pyboy, base + i) for i in range(4)]


def get_tile_in_front(pyboy: "PyBoy") -> int:
    """
    Get the tile ID that the player is currently facing.

    This is used to detect whether the player is facing a cuttable tree,
    water tile, or boulder that requires an HM field move.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Tile ID (0-255) of the tile in front of the player.

    Example:
        >>> tile = get_tile_in_front(pyboy)
        >>> if tile in CUTTABLE_TREE_TILES:
        ...     print("Facing a cuttable tree!")

    Notes:
        - Only meaningful in the overworld (not in menus or battle)
        - The tile value depends on the current tileset
        - Used by HM automation to trigger Cut/Surf/Strength
    """
    return read_byte(pyboy, ADDR_TILE_IN_FRONT)


def party_has_move(pyboy: "PyBoy", move_id: int) -> bool:
    """
    Check if any Pokemon in the party knows a specific move.

    Scans all party Pokemon's move slots for the given move ID.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        move_id: The move ID to search for (e.g., MOVE_CUT = 0x0F).

    Returns:
        True if any party Pokemon knows the move, False otherwise.

    Example:
        >>> if party_has_move(pyboy, MOVE_CUT):
        ...     print("A party Pokemon knows Cut!")
    """
    count = get_party_count(pyboy)
    for slot in range(count):
        if move_id in get_party_moves(pyboy, slot):
            return True
    return False


# =============================================================================
# MEMORY ADDRESS CONSTANTS - BATTLE DETAIL (AGENT MODULE)
# =============================================================================
# Extended battle addresses used by the modular agent system for tactical
# decision-making. These provide detailed information about both the player's
# active Pokemon and the enemy Pokemon during battles.

# Player's Active Pokemon Species (in battle copy)
# Address: 0xD014
# Size: 1 byte (internal species ID, NOT Pokedex number)
# Source: pokered disassembly (wBattleMonSpecies)
ADDR_BATTLE_PLAYER_SPECIES = 0xD014

# Player's Active Pokemon Status Condition (in battle)
# Address: 0xD018
# Size: 1 byte
# Values: 0=none, bit 0-2=sleep turns, bit 3=poison, bit 4=burn,
#         bit 5=freeze, bit 6=paralysis
# Source: pokered disassembly (wBattleMonStatus)
ADDR_BATTLE_PLAYER_STATUS = 0xD018

# Player's Active Pokemon Type 1 (in battle)
# Address: 0xD019
# Size: 1 byte (type index: 0=Normal through 0x1A=Dragon)
# Source: pokered disassembly (wBattleMonType1)
# Note: Game uses non-contiguous type IDs, NOT 0-14 sequential
ADDR_BATTLE_PLAYER_TYPE1 = 0xD019

# Player's Active Pokemon Type 2 (in battle)
# Address: 0xD01A
# Size: 1 byte (same encoding as Type 1)
# Source: pokered disassembly (wBattleMonType2)
ADDR_BATTLE_PLAYER_TYPE2 = 0xD01A

# Player's Active Pokemon Move IDs (in battle copy)
# Address: 0xD01C
# Size: 4 bytes (1 byte per move slot)
# Source: pokered disassembly (wBattleMonMoves)
ADDR_BATTLE_PLAYER_MOVES = 0xD01C

# Player's Active Pokemon Level (in battle)
# Address: 0xD022
# Size: 1 byte (1-100)
# Source: pokered disassembly (wBattleMonLevel)
ADDR_BATTLE_PLAYER_LEVEL = 0xD022

# Player's Active Pokemon PP Values (in battle copy)
# Address: 0xD02D
# Size: 4 bytes (1 byte per move, upper 2 bits = PP Ups)
# Source: pokered disassembly (wBattleMonPP)
# Note: Actual PP = value & 0x3F (mask off PP Up bits)
ADDR_BATTLE_PLAYER_PP = 0xD02D

# Enemy Pokemon Species
# Address: 0xCFE5
# Size: 1 byte (internal species ID)
# Source: pokered disassembly (wEnemyMonSpecies2)
ADDR_BATTLE_ENEMY_SPECIES = 0xCFE5

# Enemy Pokemon Status Condition
# Address: 0xCFE9
# Size: 1 byte (same encoding as player status)
# Source: pokered disassembly (wEnemyMonStatus)
ADDR_BATTLE_ENEMY_STATUS = 0xCFE9

# Enemy Pokemon Type 1
# Address: 0xCFEA
# Size: 1 byte
# Source: pokered disassembly (wEnemyMonType1)
ADDR_BATTLE_ENEMY_TYPE1 = 0xCFEA

# Enemy Pokemon Type 2
# Address: 0xCFEB
# Size: 1 byte
# Source: pokered disassembly (wEnemyMonType2)
ADDR_BATTLE_ENEMY_TYPE2 = 0xCFEB

# Enemy Pokemon Move IDs
# Address: 0xCFED
# Size: 4 bytes (1 byte per move slot)
# Source: pokered disassembly (wEnemyMonMoves)
ADDR_BATTLE_ENEMY_MOVES = 0xCFED

# Enemy Pokemon Level
# Address: 0xCFF3
# Size: 1 byte (1-100)
# Source: pokered disassembly (wEnemyMonLevel)
ADDR_BATTLE_ENEMY_LEVEL = 0xCFF3

# Player Stat Modifiers (in-battle stages)
# Address: 0xCD1A
# Size: 7 bytes: attack, defense, speed, special, accuracy, evasion, (unused)
# Values: 1-13 where 7 = neutral (no modification)
# Source: pokered disassembly (wPlayerMonStatMods)
ADDR_PLAYER_STAT_MODS = 0xCD1A

# Enemy Stat Modifiers (in-battle stages)
# Address: 0xCD2E
# Size: 7 bytes (same layout as player)
# Source: pokered disassembly (wEnemyMonStatMods)
ADDR_ENEMY_STAT_MODS = 0xCD2E

# Player Facing Direction (sprite movement byte)
# Address: 0xC109
# Size: 1 byte
# Values: 0x00=down, 0x04=up, 0x08=left, 0x0C=right
# Source: pokered disassembly (wSpritePlayerStateData1FacingDirection)
# Note: This is the sprite state byte, not a simple 0-3 enum
ADDR_FACING_DIRECTION = 0xC109


# =============================================================================
# GAME TYPE ID CONVERSION TABLE
# =============================================================================
# Pokemon Red uses non-contiguous type IDs internally. This mapping converts
# the game's internal type IDs to our sequential 0-14 index used in the
# type effectiveness chart.
#
# Source: pokered disassembly (constants/type_constants.asm)
GAME_TYPE_TO_INDEX: dict[int, int] = {
    0x00: 0,   # NORMAL
    0x14: 1,   # FIRE
    0x15: 2,   # WATER
    0x17: 3,   # ELECTRIC
    0x16: 4,   # GRASS
    0x19: 5,   # ICE
    0x01: 6,   # FIGHTING
    0x03: 7,   # POISON
    0x04: 8,   # GROUND
    0x02: 9,   # FLYING
    0x18: 10,  # PSYCHIC
    0x07: 11,  # BUG
    0x05: 12,  # ROCK
    0x08: 13,  # GHOST
    0x1A: 14,  # DRAGON
}


# =============================================================================
# GAME STATE READING FUNCTIONS - BATTLE DETAIL (AGENT MODULE)
# =============================================================================
# Extended battle-reading functions for the modular agent system.
# These provide detailed battle information for tactical decision-making.


def get_battle_player_pokemon(pyboy: "PyBoy") -> dict[str, int | list[int]]:
    """
    Get detailed info about the player's active Pokemon in battle.

    Reads the battle copy of the player's Pokemon data, which includes
    species, types, moves, PP, level, HP, and status.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Dictionary with battle Pokemon data:
        - species: Internal species ID (1-190)
        - hp: Current HP (16-bit)
        - max_hp: Maximum HP (16-bit)
        - level: Pokemon level (1-100)
        - type1: Game-internal type ID for primary type
        - type2: Game-internal type ID for secondary type
        - moves: List of 4 move IDs (0 = empty)
        - pp: List of 4 PP values (masked to 6 bits)
        - status: Status condition byte

    Notes:
        - Only valid during battles (check is_in_battle() first)
        - PP values have upper 2 bits masked off (PP Up encoding)
        - Types use game-internal IDs, convert with GAME_TYPE_TO_INDEX
    """
    return {
        "species": read_byte(pyboy, ADDR_BATTLE_PLAYER_SPECIES),
        "hp": read_word(pyboy, ADDR_PLAYER_HP),
        "max_hp": read_word(pyboy, ADDR_PLAYER_MAX_HP),
        "level": read_byte(pyboy, ADDR_BATTLE_PLAYER_LEVEL),
        "type1": read_byte(pyboy, ADDR_BATTLE_PLAYER_TYPE1),
        "type2": read_byte(pyboy, ADDR_BATTLE_PLAYER_TYPE2),
        "moves": [read_byte(pyboy, ADDR_BATTLE_PLAYER_MOVES + i) for i in range(4)],
        "pp": [read_byte(pyboy, ADDR_BATTLE_PLAYER_PP + i) & 0x3F for i in range(4)],
        "status": read_byte(pyboy, ADDR_BATTLE_PLAYER_STATUS),
    }


def get_battle_enemy_pokemon(pyboy: "PyBoy") -> dict[str, int | list[int]]:
    """
    Get detailed info about the enemy Pokemon in battle.

    Reads the enemy Pokemon's battle data including species, types,
    moves, level, HP, and status.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Dictionary with enemy Pokemon data (same structure as player).

    Notes:
        - Only valid during battles (check is_in_battle() first)
        - Enemy PP is not easily accessible in Gen 1 memory
        - Types use game-internal IDs, convert with GAME_TYPE_TO_INDEX
    """
    return {
        "species": read_byte(pyboy, ADDR_BATTLE_ENEMY_SPECIES),
        "hp": read_word(pyboy, ADDR_ENEMY_HP),
        "max_hp": read_word(pyboy, ADDR_ENEMY_MAX_HP),
        "level": read_byte(pyboy, ADDR_BATTLE_ENEMY_LEVEL),
        "type1": read_byte(pyboy, ADDR_BATTLE_ENEMY_TYPE1),
        "type2": read_byte(pyboy, ADDR_BATTLE_ENEMY_TYPE2),
        "moves": [read_byte(pyboy, ADDR_BATTLE_ENEMY_MOVES + i) for i in range(4)],
        "pp": [],  # Enemy PP not reliably readable in Gen 1
        "status": read_byte(pyboy, ADDR_BATTLE_ENEMY_STATUS),
    }


def get_stat_modifiers(pyboy: "PyBoy", is_player: bool = True) -> list[int]:
    """
    Get in-battle stat modification stages.

    During battle, stat-modifying moves (Swords Dance, Growl, etc.) change
    stat stages stored in memory. Stages range from 1 (minimum, -6) to
    13 (maximum, +6), with 7 being neutral (no modification).

    Args:
        pyboy: PyBoy emulator instance with the game loaded.
        is_player: True for player's stat mods, False for enemy's.

    Returns:
        List of 6 stat stages [attack, defense, speed, special, accuracy, evasion].
        Each value is 1-13, where 7 = neutral.

    Notes:
        - Stat stages are reset when a Pokemon switches out
        - The actual multiplier is: stage/2 for stages 1-7, stage-1 for 8-13
        - Only valid during battles
    """
    base = ADDR_PLAYER_STAT_MODS if is_player else ADDR_ENEMY_STAT_MODS
    return [read_byte(pyboy, base + i) for i in range(6)]


def get_battle_type(pyboy: "PyBoy") -> int:
    """
    Get the type of the current battle.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Battle type: 0 = wild, 1 = trainer, 2 = scripted.
        Returns 0 if not in battle.

    Notes:
        - Wild battles allow running, trainer battles do not
        - The agent uses this to decide run vs fight strategy
    """
    if not is_in_battle(pyboy):
        return 0
    return read_byte(pyboy, ADDR_BATTLE_TYPE)


def get_facing_direction(pyboy: "PyBoy") -> int:
    """
    Get the direction the player sprite is currently facing.

    Args:
        pyboy: PyBoy emulator instance with the game loaded.

    Returns:
        Direction as 0-3: 0=down, 1=up, 2=left, 3=right.

    Notes:
        - The game stores sprite direction as 0x00/0x04/0x08/0x0C
        - We convert to a simple 0-3 index for convenience
        - Used by the navigator to determine movement direction
    """
    raw = read_byte(pyboy, ADDR_FACING_DIRECTION)
    # Convert sprite direction byte to 0-3 index
    # 0x00=down(0), 0x04=up(1), 0x08=left(2), 0x0C=right(3)
    return raw >> 2
