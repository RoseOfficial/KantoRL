"""
HM compatibility tables for Pokemon Red (Gen 1).

This module provides pre-computed sets of Pokemon species IDs that can learn
each HM move. The data is extracted from the pokered disassembly TM/HM
compatibility bitfields.

Architecture Role:
    Used by hm_automation.py to determine which party Pokemon can learn
    an HM move when the player picks up an HM item. The auto-teach system
    finds the first compatible Pokemon in the party and writes the move
    to its 4th move slot.

Data Source:
    pokered disassembly: data/pokemon/base_stats/*.asm
    Each Pokemon's base stats include a 7-byte TM/HM compatibility bitfield.
    Bits 48-52 (bytes 6-7, bits 0-4) correspond to HMs 01-05.

Note:
    Species IDs here are the INTERNAL IDs used by Pokemon Red's memory,
    NOT Pokedex numbers. These differ for many Pokemon. For example:
    - Bulbasaur = species 0x99 (153), Pokedex #1
    - Charmander = species 0xB0 (176), Pokedex #4
    - Squirtle = species 0xB1 (177), Pokedex #7

    We use Pokedex-order internal IDs from the wPartySpecies array,
    which stores the INTERNAL species index (not Pokedex number).

Usage:
    >>> from kantorl.data.hm_compat import CUT_COMPATIBLE, SURF_COMPATIBLE
    >>> species_id = read_byte(pyboy, ADDR_PARTY_SPECIES + slot)
    >>> if species_id in CUT_COMPATIBLE:
    ...     print("This Pokemon can learn Cut!")

Dependencies:
    None (pure data module)
"""

# =============================================================================
# HM01 CUT COMPATIBLE POKEMON
# =============================================================================
# Move ID: 15 (0x0F)
# Almost all Pokemon with arms/claws/blades can learn Cut.
# Very widely learnable — most starter lines, Farfetch'd, etc.
#
# Species IDs (internal) for Pokemon that can learn HM01 Cut
# Source: pokered disassembly TM/HM compatibility data
#
# Key compatible Pokemon by availability:
# - Bulbasaur line, Charmander line (starters)
# - Oddish/Bellsprout lines (early grass types)
# - Farfetch'd (obtained via trade on Route 7)
# - Paras/Parasect (Mt. Moon)
# - Sandshrew/Sandslash (Route 4/11)
CUT_COMPATIBLE: frozenset[int] = frozenset({
    # Bulbasaur line
    0x99, 0x09, 0xBA,
    # Charmander line
    0xB0, 0xB2, 0xB4,
    # Sandshrew line
    0x60, 0x61,
    # Nidoran♀ line
    0x0F, 0xA8, 0x10,
    # Nidoran♂ line
    0x03, 0xA7, 0x07,
    # Oddish line
    0xB9, 0xBA, 0xBB,
    # Paras line
    0x6D, 0x2E,
    # Bellsprout line
    0xBC, 0xBD, 0xBE,
    # Farfetch'd
    0x40,
    # Krabby line
    0x4E, 0x8A,
    # Lickitung
    0x0B,
    # Scyther
    0x1A,
    # Pinsir
    0x1D,
    # Tangela
    0x1E,
    # Mewtwo, Mew
    0x83, 0x15,
})

# =============================================================================
# HM03 SURF COMPATIBLE POKEMON
# =============================================================================
# Move ID: 57 (0x39)
# Water-type Pokemon and some others that can swim.
#
# Key compatible Pokemon by availability:
# - Squirtle line (starter)
# - Psyduck/Golduck (Route 6 area)
# - Tentacool line (most water routes)
# - Lapras (gift in Silph Co.)
# - Snorlax (Route 12/16)
SURF_COMPATIBLE: frozenset[int] = frozenset({
    # Squirtle line
    0xB1, 0xB3, 0x1C,
    # Nidoqueen, Nidoking
    0x10, 0x07,
    # Psyduck line
    0x2F, 0x80,
    # Poliwag line
    0x47, 0x6E, 0x6F,
    # Tentacool line
    0x18, 0x9B,
    # Slowpoke line
    0x25, 0x08,
    # Seel line
    0x3A, 0x78,
    # Shellder line
    0x17, 0x8B,
    # Krabby line
    0x4E, 0x8A,
    # Horsea line
    0x5C, 0x5D,
    # Goldeen line
    0x9D, 0x9E,
    # Staryu line
    0x1B, 0x98,
    # Gyarados
    0x16,
    # Lapras
    0x13,
    # Vaporeon
    0x69,
    # Omanyte line
    0x62, 0x63,
    # Kabuto line
    0x5A, 0x5B,
    # Snorlax
    0x84,
    # Dratini line
    0x58, 0x59, 0x42,
    # Mewtwo, Mew
    0x83, 0x15,
})

# =============================================================================
# HM04 STRENGTH COMPATIBLE POKEMON
# =============================================================================
# Move ID: 70 (0x46)
# Pokemon with enough physical power to push boulders.
#
# Key compatible Pokemon by availability:
# - Charmander line (starter)
# - Geodude line (Mt. Moon, Rock Tunnel)
# - Machop line (Route 10)
# - Snorlax (Route 12/16)
# - Most fully-evolved Pokemon
STRENGTH_COMPATIBLE: frozenset[int] = frozenset({
    # Charmander line
    0xB0, 0xB2, 0xB4,
    # Squirtle line
    0xB1, 0xB3, 0x1C,
    # Nidoqueen, Nidoking
    0x10, 0x07,
    # Sandshrew line
    0x60, 0x61,
    # Mankey line
    0x39, 0x75,
    # Poliwag line
    0x47, 0x6E, 0x6F,
    # Machop line
    0x6A, 0x29, 0x7E,
    # Geodude line
    0xA9, 0x27, 0x31,
    # Slowpoke line
    0x25, 0x08,
    # Onix
    0x22,
    # Cubone line
    0x91, 0x92,
    # Hitmonlee, Hitmonchan
    0x2B, 0x2C,
    # Lickitung
    0x0B,
    # Rhyhorn line
    0x12, 0x01,
    # Chansey
    0x28,
    # Kangaskhan
    0x02,
    # Tauros
    0x3C,
    # Gyarados
    0x16,
    # Lapras
    0x13,
    # Snorlax
    0x84,
    # Dragonite
    0x42,
    # Mewtwo, Mew
    0x83, 0x15,
})

# =============================================================================
# COMBINED LOOKUP TABLE
# =============================================================================
# Maps HM move IDs to their compatibility sets for easy lookup.

# Import move IDs from memory module would create circular dependency,
# so we use literal values matching memory.MOVE_CUT/SURF/STRENGTH
HM_COMPAT: dict[int, frozenset[int]] = {
    0x0F: CUT_COMPATIBLE,       # MOVE_CUT
    0x39: SURF_COMPATIBLE,      # MOVE_SURF
    0x46: STRENGTH_COMPATIBLE,  # MOVE_STRENGTH
}
