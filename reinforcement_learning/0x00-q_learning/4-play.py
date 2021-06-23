#!/usr/bin/env python3
"""
Function that has the trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    You should always exploit the Q-table.
    Returns: the total rewards for the episode.
    """
    env.reset()
    state = env.reset()
    done = False
    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward
        # given that state
        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        if done:
            # Here, we decide to only print the last state
            # (to see if our agent is on the goal or fall into an hole)
            env.render()

            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
    env.close()

    return reward
