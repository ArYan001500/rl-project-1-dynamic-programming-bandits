# RL project 1: Dynamic programming and bandits

This project covers tabular reinforcement-learning algorithms in small environments.

## Contents

- `src/gridworld_policy_iteration.py`: policy iteration for a 4x4 Gridworld with terminal states and `gamma = 0.9`.
- `src/ten_armed_bandit.py`: epsilon-greedy and UCB comparison for the ten-armed bandit problem.
- `src/gridworld_value_matrix.py`: 5x5 Gridworld value-matrix solution using a Markov reward process and a linear-system solve.

## Results from the project notes

In the bandit experiment, UCB with `c = 2` produced higher and more stable average reward than epsilon-greedy with `epsilon = 0.1` across 2000 runs. The Gridworld work connects Bellman equations, policy evaluation, policy improvement, and matrix-based value solving.
