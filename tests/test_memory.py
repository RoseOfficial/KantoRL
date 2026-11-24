"""Party reads must follow Gen 1's 44-byte struct layout and big-endian HP.

The old contiguous reads returned max-HP and stat bytes as levels, which
showed up as fake level-89 party members in evaluation.
"""

from kantorl import memory


class FakeMemory:
    """Sparse byte-addressable memory for a fake PyBoy."""

    def __init__(self) -> None:
        self.values: dict[int, int] = {}

    def __getitem__(self, addr: int) -> int:
        return self.values.get(addr, 0)


class FakePyBoy:
    """Minimal PyBoy stand-in exposing only ``memory[addr]``."""

    def __init__(self) -> None:
        self.memory = FakeMemory()


def _write_word_be(fake: FakePyBoy, addr: int, value: int) -> None:
    """Write a 16-bit value high byte first, as Gen 1 stores HP."""
    fake.memory.values[addr] = (value >> 8) & 0xFF
    fake.memory.values[addr + 1] = value & 0xFF


def _write_party_member(
    fake: FakePyBoy,
    slot: int,
    *,
    level: int,
    current_hp: int,
    max_hp: int,
) -> None:
    """Place one party member at its 44-byte struct offset."""
    base = memory.ADDR_PARTY_HP + slot * memory.PARTY_MON_STRIDE
    _write_word_be(fake, base, current_hp)
    max_hp_addr = memory.ADDR_PARTY_MAX_HP + slot * memory.PARTY_MON_STRIDE
    _write_word_be(fake, max_hp_addr, max_hp)
    level_addr = memory.ADDR_PARTY_LEVEL + slot * memory.PARTY_MON_STRIDE
    fake.memory.values[level_addr] = level


def test_read_word_is_big_endian() -> None:
    fake = FakePyBoy()
    fake.memory.values[0x1000] = 0x12
    fake.memory.values[0x1001] = 0x34

    assert memory.read_word(fake, 0x1000) == 0x1234


def test_party_levels_use_the_44_byte_struct_stride() -> None:
    fake = FakePyBoy()
    fake.memory.values[memory.ADDR_PARTY_COUNT] = 3
    _write_party_member(fake, 0, level=15, current_hp=25, max_hp=30)
    _write_party_member(fake, 1, level=12, current_hp=20, max_hp=25)
    _write_party_member(fake, 2, level=10, current_hp=18, max_hp=20)

    assert memory.get_party_levels(fake) == [15, 12, 10]


def test_party_hp_uses_stride_and_big_endian_words() -> None:
    fake = FakePyBoy()
    fake.memory.values[memory.ADDR_PARTY_COUNT] = 3
    _write_party_member(fake, 0, level=15, current_hp=25, max_hp=30)
    _write_party_member(fake, 1, level=12, current_hp=20, max_hp=25)
    _write_party_member(fake, 2, level=10, current_hp=18, max_hp=20)

    assert memory.get_party_hp(fake) == [(25, 30), (20, 25), (18, 20)]
    assert memory.get_total_party_hp(fake) == (63, 75)


def test_party_reads_skip_empty_slots_but_keep_fainted() -> None:
    fake = FakePyBoy()
    fake.memory.values[memory.ADDR_PARTY_COUNT] = 3
    # Slot 0 is fainted but real: current HP 0, max HP 30.
    _write_party_member(fake, 0, level=15, current_hp=0, max_hp=30)
    # Slot 1 is an empty/phantom slot: every byte is zero.
    # Slot 2 is a healthy party member.
    _write_party_member(fake, 2, level=10, current_hp=5, max_hp=20)

    assert memory.get_party_levels(fake) == [15, 10]
    assert memory.get_party_hp(fake) == [(0, 30), (5, 20)]
    assert memory.get_total_party_hp(fake) == (5, 50)
