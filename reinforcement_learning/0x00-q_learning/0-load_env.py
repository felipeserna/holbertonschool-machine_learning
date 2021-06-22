#!/usr/bin/env python3
"""
Loads the pre-made FrozenLakeEnv environment
from OpenAI’s gym
"""
import numpy as np
import gym
import random
import time


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Returns: the environment
    """
    # Creating The Environment
    env = gym.make("FrozenLake-v0", desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
