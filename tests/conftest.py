"""
Shared pytest fixtures for KantoRL test suite.

This module provides common test fixtures used across all test modules,
including mocked PyBoy instances, default configurations, and test utilities.

Fixtures:
    mock_pyboy: Mocked PyBoy emulator for testing without ROMs
    default_config: Default KantoConfig instance
    mock_env: KantoRedEnv with mocked PyBoy
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch


# =============================================================================
# PYBOY MOCKING FIXTURES
# =============================================================================

@pytest.fixture
def mock_pyboy() -> MagicMock:
    """
    Create a mocked PyBoy instance for testing.

    This fixture provides a fully mocked PyBoy emulator that can be used
    to test memory reading, screen capture, and other PyBoy-dependent
    functionality without requiring an actual ROM file.

    Returns:
        MagicMock configured to behave like a PyBoy instance with:
        - memory: Mocked memory interface returning 0 by default
        - screen.ndarray: 144x160x4 uint8 array (RGBA format)
        - tick(): No-op method
        - button(): No-op method for button presses

    Example:
        >>> def test_memory_read(mock_pyboy):
        ...     mock_pyboy.memory.__getitem__.return_value = 42
        ...     assert mock_pyboy.memory[0xD163] == 42
    """
    mock = MagicMock()

    # Mock memory interface
    # Default return value of 0 for any memory address
    mock.memory = MagicMock()
    mock.memory.__getitem__ = Mock(return_value=0)

    # Mock screen interface
    # Returns a blank 144x160 RGBA screen (GameBoy resolution)
    mock.screen = MagicMock()
    mock.screen.ndarray = np.zeros((144, 160, 4), dtype=np.uint8)

    # Mock tick method (advances emulator by one frame)
    mock.tick = Mock(return_value=None)

    # Mock button method (sends button input)
    mock.button = Mock(return_value=None)

    return mock


@pytest.fixture
def mock_pyboy_with_party(mock_pyboy: MagicMock) -> MagicMock:
    """
    Create a mocked PyBoy with a configured party.

    Extends the base mock_pyboy fixture with memory values that simulate
    a party of Pokemon with specific levels and HP values.

    Args:
        mock_pyboy: Base mocked PyBoy fixture.

    Returns:
        MagicMock with memory configured for:
        - Party count: 3 Pokemon
        - Party levels: [15, 12, 10, 0, 0, 0]
        - Party HP: Full health for all party members

    Notes:
        Memory addresses used:
        - 0xD163: Party count (wPartyCount)
        - 0xD18C, 0xD1B8, etc.: Pokemon level addresses
        - 0xD16C, 0xD198, etc.: Pokemon current HP addresses
    """
    # Party count memory address
    PARTY_COUNT_ADDR = 0xD163

    # Party level addresses (offset by 0x2C for each party slot)
    LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

    # Current HP addresses (high byte, offset by 0x2C for each party slot)
    HP_ADDRS = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]

    # Configure memory return values
    def memory_getter(addr: int) -> int:
        if addr == PARTY_COUNT_ADDR:
            return 3  # 3 Pokemon in party
        for i, level_addr in enumerate(LEVEL_ADDRS):
            if addr == level_addr:
                return [15, 12, 10, 0, 0, 0][i]
        for i, hp_addr in enumerate(HP_ADDRS):
            if addr == hp_addr:
                return 100 if i < 3 else 0  # Full HP for party members
        return 0

    mock_pyboy.memory.__getitem__ = Mock(side_effect=memory_getter)
    return mock_pyboy


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """
    Create a default KantoConfig instance for testing.

    Returns:
        KantoConfig with default values as specified in config.py.

    Example:
        >>> def test_config_values(default_config):
        ...     assert default_config.frame_skip == 24
    """
    from kantorl.config import KantoConfig
    return KantoConfig()


@pytest.fixture
def custom_config():
    """
    Create a KantoConfig with custom values for testing edge cases.

    Returns:
        KantoConfig with modified values:
        - frame_skip: 1 (for detailed frame-by-frame testing)
        - max_steps: 100 (for quick episode termination)
        - headless: True (no display)

    Example:
        >>> def test_short_episode(custom_config, mock_env):
        ...     # Episode will terminate after 100 steps
        ...     pass
    """
    from kantorl.config import KantoConfig
    return KantoConfig(
        frame_skip=1,
        max_steps=100,
        headless=True,
    )


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def mock_env(mock_pyboy: MagicMock, default_config):
    """
    Create a KantoRedEnv with mocked PyBoy for testing.

    This fixture patches the PyBoy constructor to return the mock_pyboy
    fixture, allowing environment testing without a ROM file.

    Args:
        mock_pyboy: Mocked PyBoy instance.
        default_config: Default configuration.

    Yields:
        KantoRedEnv instance with mocked emulator.

    Notes:
        The environment is properly closed after the test completes.

    Example:
        >>> def test_env_reset(mock_env):
        ...     obs, info = mock_env.reset()
        ...     assert 'screens' in obs
    """
    from kantorl.env import KantoRedEnv

    with patch('kantorl.env.PyBoy', return_value=mock_pyboy):
        env = KantoRedEnv(rom_path='fake.gb', config=default_config)
        yield env
        env.close()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def sample_game_state() -> dict:
    """
    Create a sample game state dictionary for reward testing.

    Returns:
        Dictionary containing typical game state values:
        - map_id: Current map identifier
        - x, y: Player coordinates
        - badges: Number of badges obtained
        - party_level: Sum of party Pokemon levels
        - party_hp: Total current HP
        - events: Set of triggered event flags

    Example:
        >>> def test_reward_calculation(sample_game_state):
        ...     reward = calculate_reward(sample_game_state, previous_state)
    """
    return {
        'map_id': 0,  # Pallet Town
        'x': 5,
        'y': 6,
        'badges': 0,
        'party_level': 15,
        'party_hp': 100,
        'party_max_hp': 100,
        'events': set(),
    }


@pytest.fixture
def random_seed() -> int:
    """
    Provide a fixed random seed for reproducible tests.

    Returns:
        Fixed seed value (42) for numpy and other RNG.

    Notes:
        Use this fixture to ensure tests are deterministic.
    """
    return 42
