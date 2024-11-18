# Modificaciones importantes:
# 1. Exploración extendida (epsilon_decay ajustado).
# 2. Recompensas dinámicas basadas en la distancia a la meta.
# 3. Más episodios para mejorar el aprendizaje.
# 4. Visualización opcional de progreso durante el entrenamiento.

import numpy as np
import matplotlib.pyplot as plt
import random
import sys

sys.setrecursionlimit(3000)  # Adjust as needed

# ---------------- Generar Laberinto ----------------
def generate_maze(size=100):
    maze = np.ones((size, size), dtype=int)
    def carve_passages_from(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size and 0 < ny < size and maze[nx, ny] == 1:
                maze[x + dx // 2, y + dy // 2] = 0
                maze[nx, ny] = 0
                carve_passages_from(nx, ny)
    maze[1, 1] = 0
    carve_passages_from(1, 1)
    while True:
        goal_x, goal_y = random.randint(1, size - 2), random.randint(1, size - 2)
        if maze[goal_x, goal_y] == 0:
            maze[goal_x, goal_y] = 2
            break
    return maze, (goal_x, goal_y)

# Generar laberinto y meta
maze, goal = generate_maze(size=100)
print(f"Labyrinth Goal at: {goal}")

# ---------------- Parámetros y Validación ----------------
actions = ['up', 'down', 'left', 'right']

def is_valid_move(maze, x, y):
    if x < 0 or y < 0 or x >= maze.shape[0] or y >= maze.shape[1]:
        return False
    return maze[x, y] != 1

def move(maze, position, action):
    x, y = position
    if action == 'up' and is_valid_move(maze, x-1, y):
        return x-1, y
    elif action == 'down' and is_valid_move(maze, x+1, y):
        return x+1, y
    elif action == 'left' and is_valid_move(maze, x, y-1):
        return x, y-1
    elif action == 'right' and is_valid_move(maze, x, y+1):
        return x, y+1
    return x, y

# ---------------- Implementación del Algoritmo Q-Learning ----------------
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.998  # Decadencia más lenta para mayor exploración
epsilon_min = 0.1
max_episodes = 3000  # Aumentar episodios
max_steps_per_episode = 2000

q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))

def train_agent(maze, goal, max_episodes):
    global epsilon
    for episode in range(max_episodes):
        x, y = 1, 1
        steps = 0

        while maze[x, y] != 2 and steps < max_steps_per_episode:
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(q_table[x, y])]

            new_x, new_y = move(maze, (x, y), action)

            # Recompensas ajustadas
            if (new_x, new_y) == goal:
                reward = 100
            elif (new_x, new_y) == (x, y):
                reward = -50
            else:
                distance_to_goal = abs(goal[0] - new_x) + abs(goal[1] - new_y)
                reward = -1 - 0.1 * distance_to_goal  # Penalización basada en distancia

            best_next_action = np.max(q_table[new_x, new_y])
            q_table[x, y, actions.index(action)] += alpha * (
                reward + gamma * best_next_action - q_table[x, y, actions.index(action)]
            )

            x, y = new_x, new_y
            steps += 1

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{max_episodes} completed.")

train_agent(maze, goal, max_episodes)

# ---------------- Evaluación del Modelo ----------------
def evaluate_agent(maze, start, goal):
    path = [start]
    x, y = start
    steps = 0

    while maze[x, y] != 2 and steps < max_steps_per_episode:
        action = actions[np.argmax(q_table[x, y])]
        new_x, new_y = move(maze, (x, y), action)
        if (new_x, new_y) == (x, y):
            break
        x, y = new_x, new_y
        path.append((x, y))
        steps += 1

    return path

# ---------------- Visualización ----------------
def plot_path(maze, path, goal):
    plt.figure(figsize=(12, 12))
    plt.imshow(maze, cmap=plt.cm.binary, origin="upper")
    for (x, y) in path:
        plt.scatter(y, x, c='red', s=5)
    plt.scatter(*goal[::-1], c='blue', s=100, label='Goal')
    plt.title("Recorrido del Agente en el Laberinto Realista (100x100)")
    plt.legend()
    plt.show()

start = (1, 1)
path = evaluate_agent(maze, start, goal)
plot_path(maze, path, goal)
