#!/usr/bin/env python3
"""
Performs SARSA(λ)
"""
import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, the updated Q table
    """
