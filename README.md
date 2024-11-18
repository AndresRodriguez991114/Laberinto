# Q-Learning Maze Solver

This project implements a Q-Learning algorithm to train an agent to navigate a 100x100 maze towards a defined goal.

## Description

- **Maze:** A 100x100 grid where:
  - `0`: Free path
  - `1`: Obstacles
  - `2`: Goal
- **Actions:** Up, down, left, right.
- **Algorithm:** Q-Learning with parameters:
  - Alpha (learning rate): 0.1
  - Gamma (discount factor): 0.9
  - Epsilon (exploration factor): 0.1
- **Training:** 1000 episodes to learn an optimal policy.

## Usage

1. Run `qlearning_maze.py` to generate a maze, train the agent, and visualize the path.

## Results

The trained agent learns to navigate efficiently from a random start point to the goal while avoiding obstacles.
