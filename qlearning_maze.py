import numpy as np
import matplotlib.pyplot as plt

# Crear laberinto 100x100
def create_maze(size=100):
    maze = np.zeros((size, size), dtype=int)

    # Añadir obstáculos
    np.random.seed(42)  # Para reproducibilidad
    for _ in range(size * 10):  # Agregar obstáculos aleatorios
        x, y = np.random.randint(0, size, size=2)
        maze[x, y] = 1

    # Definir la meta
    goal_x, goal_y = np.random.randint(0, size, size=2)
    maze[goal_x, goal_y] = 2

    return maze, (goal_x, goal_y)

# Crear laberinto y meta
maze, goal = create_maze()
print(f"Labyrinth Goal at: {goal}")

# Acciones: 0: arriba, 1: abajo, 2: izquierda, 3: derecha
actions = ['up', 'down', 'left', 'right']

# Validación del movimiento
def is_valid_move(maze, x, y):
    if x < 0 or y < 0 or x >= maze.shape[0] or y >= maze.shape[1]:
        return False
    return maze[x, y] != 1

# Realizar un movimiento
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
    return x, y  # Movimiento inválido regresa la posición original

# Configuración de parámetros
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.9  # Factor de descuento
epsilon = 0.1  # Exploración-explotación
episodes = 1000  # Número de episodios
size = 100

# Crear tabla Q
q_table = np.zeros((size, size, len(actions)))

# Entrenamiento Q-Learning
def train_agent(maze, goal, episodes=1000):
    for episode in range(episodes):
        # Inicializar posición aleatoria
        x, y = np.random.randint(0, maze.shape[0], size=2)

        while maze[x, y] != 2:  # Hasta alcanzar el objetivo
            # Elegir acción (epsilon-greedy)
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = actions[np.argmax(q_table[x, y])]

            # Realizar acción
            new_x, new_y = move(maze, (x, y), action)
            reward = 1 if (new_x, new_y) == goal else -0.1

            # Actualizar tabla Q
            best_next_action = np.max(q_table[new_x, new_y])
            q_table[x, y, actions.index(action)] = q_table[x, y, actions.index(action)] + \
                alpha * (reward + gamma * best_next_action - q_table[x, y, actions.index(action)])

            # Actualizar posición
            x, y = new_x, new_y

train_agent(maze, goal, episodes)

# Probar agente con un límite de pasos para evitar bucles infinitos
def evaluate_agent(maze, start, goal, max_steps=1000):
    path = [start]
    x, y = start
    steps = 0

    while maze[x, y] != 2 and steps < max_steps:
        action = actions[np.argmax(q_table[x, y])]
        new_x, new_y = move(maze, (x, y), action)
        
        # Si el agente no se mueve, puede estar atrapado
        if (new_x, new_y) == (x, y):
            break

        x, y = new_x, new_y
        path.append((x, y))
        steps += 1

    return path

# Generar camino
start = (0, 0)
path = evaluate_agent(maze, start, goal)

# Visualizar el recorrido
def plot_path(maze, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='gray')
    for (x, y) in path:
        plt.scatter(y, x, c='red', s=1)
    plt.title("Recorrido del Agente")
    plt.show()

plot_path(maze, path)
