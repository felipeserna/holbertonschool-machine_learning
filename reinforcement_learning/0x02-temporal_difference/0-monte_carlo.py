#!/usr/bin/env python3
"""
Performs the Monte Carlo algorithm
"""
import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Returns: V, the updated value estimate
    """
