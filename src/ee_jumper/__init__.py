"""Minimal Jumper swimming environments."""

from gymnasium.envs.registration import register

register(id="EEJumperSwim-v0", entry_point="ee_jumper.env:SwimEnv")
