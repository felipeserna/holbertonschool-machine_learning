#!/usr/bin/env python3
"""
Implement the training
(by using the Monte-Carlo policy gradient algorithm - also called REINFORCE).
"""


from policy_gradient import policy_gradient
import numpy as np


def train(env, nb_episodes, alpha=0.00045, gamma=0.98, show_result=False):
    """
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    show_result:  is True, render the environment every 1000 episodes computed.
    """
    weights = np.random.rand(4, 2)
    episode = []

    for i in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and (i % 1000 == 0):
                env.render()
            action, grad = policy_gradient(state, weights)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state
            if done:
                break

        for j in range(len(grads)):
            weights += alpha * grads[j] *\
                sum([r * gamma ** r for t, r in enumerate(rewards[j:])])
        episode.append(score)
        print("{}: {}".format(i, score), end="\r", flush=False)

    return (episode)
