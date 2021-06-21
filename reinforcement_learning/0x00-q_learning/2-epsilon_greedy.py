#!/usr/bin/env python3
"""
Uses epsilon-greedy to determine the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Returns: the next action index
    """
    