#!/usr/bin/env python3
"""
Performs Q-learning
"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Returns: Q, total_rewards
    """
