from gymnasium.envs.registration import register

from .baseline import CascadeController
from .demand import DemandProfile
from .env import CompressorEnv, EnvConfig, FixedSpec, VSDSpec

register(id="CompressorSequencing-v0", entry_point="compressor_env.env:CompressorEnv")

__all__ = [
    "CascadeController",
    "CompressorEnv",
    "DemandProfile",
    "EnvConfig",
    "FixedSpec",
    "VSDSpec",
]
