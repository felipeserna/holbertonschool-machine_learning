#!/usr/bin/env python3
"""
Function that has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Returns: the total rewards for the episode
    """
    