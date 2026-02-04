"""KantoRL - The MNIST of Pokemon Red for Reinforcement Learning.

A minimal, educational RL environment for Pokemon Red.
"""

__version__ = "0.1.0"

from kantorl.config import KantoConfig
from kantorl.env import KantoRedEnv

__all__ = ["KantoConfig", "KantoRedEnv", "__version__"]
