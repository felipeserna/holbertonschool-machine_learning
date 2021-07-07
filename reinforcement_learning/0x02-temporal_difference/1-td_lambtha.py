#!/usr/bin/env python3
"""
Performs the TD(λ) algorithm
"""
import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Returns: V, the updated value estimate
    """
