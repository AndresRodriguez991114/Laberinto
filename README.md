# Q-Learning Maze Solver

This project implements a Q-Learning algorithm to train an agent to navigate a 100x100 maze towards a defined goal.

## Description

- **Maze:** A 100x100 grid where:
  - `0`: Free path
  - `1`: Obstacles (walls)
  - `2`: Goal
- **Maze Generation:** The maze is generated using a depth-first search (DFS) algorithm with a stack to ensure a realistic maze layout.
- **Actions:** The agent can move in four directions: up, down, left, right.
- **Algorithm:** Q-Learning with the following parameters:
  - **Alpha (learning rate):** 0.1
  - **Gamma (discount factor):** 0.9
  - **Epsilon (exploration factor):** Starts at 1.0 and decays to 0.1.
  - **Episodes:** 3000 training episodes.
  - **Steps per episode:** Limited to 2000 steps to avoid infinite loops.
- **Rewards:**
  - +100 for reaching the goal.
  - -50 for staying in the same position (invalid move).
  - -1 with a small penalty proportional to the Manhattan distance to the goal.

## Installation

1. Install Python 3.x and ensure the following libraries are available:
   - `numpy`
   - `matplotlib`

2. Clone the repository or download the code.

## Usage

1. Run `qlearning_maze.py` to:
   - Generate a 100x100 maze with a random goal position.
   - Train the agent using the Q-Learning algorithm.
   - Visualize the agent's path from the starting point to the goal.

2. The code will display:
   - A 100x100 maze with walls (grey), the agent's path (red dots), and the goal (blue dot).

## Results

- The agent successfully learns to navigate from the start to the goal while avoiding walls and optimizing the path.
- Visualization includes:
  - The final maze.
  - The learned path of the agent.

## Key Improvements

1. **Maze generation**:
   - Transitioned from recursion to an iterative stack-based approach to prevent recursion limit issues.
2. **Reward system**:
   - Introduced distance-based penalties to encourage the agent to move toward the goal.
3. **Exploration vs. Exploitation**:
   - Epsilon decays gradually across 3000 episodes to balance exploration and exploitation effectively.
