from collections import deque
import random
def generate_maze(size=10, obstacle_density=0.3):
    # Initialize the maze with open cells
    maze = [['.' for _ in range(size)] for _ in range(size)]

    # Add obstacles based on density
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_density:
                maze[i][j] = 'X'

    # Set starting point (S)
    start_i, start_j = random.randint(0, size - 1), random.randint(0, size - 1)
    maze[start_i][start_j] = 'S'

    # Set goal point (G) ensuring it is different from the starting point
    while True:
        goal_i, goal_j = random.randint(0, size - 1), random.randint(0, size - 1)
        if (goal_i, goal_j) != (start_i, start_j):
            maze[goal_i][goal_j] = 'G'
            break

    return maze


def breadth_first_search(maze, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    expanded_nodes = []

    while queue:
        (current_i, current_j), path = queue.popleft()

        if (current_i, current_j) == goal:
            print("Path Length:", len(path) - 1)  # Subtract 1 to exclude the starting point
            print("Path Nodes:", path)
            print("Number of Nodes Expanded:", len(expanded_nodes))
            print("Expanded Nodes:", expanded_nodes)
            return path  # Return the path if goal reached

        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))
            expanded_nodes.append((current_i, current_j))

            # Explore neighbors in the order (up, right, down, left)
            neighbors = [
                (current_i - 1, current_j),
                (current_i, current_j + 1),
                (current_i + 1, current_j),
                (current_i, current_j - 1),
            ]

            for neighbor_i, neighbor_j in neighbors:
                if (
                    0 <= neighbor_i < len(maze)
                    and 0 <= neighbor_j < len(maze[0])
                    and maze[neighbor_i][neighbor_j] != 'X'
                    and (neighbor_i, neighbor_j) not in visited
                ):
                    queue.append(((neighbor_i, neighbor_j), path + [(neighbor_i, neighbor_j)]))

    print("No path found.")
    return None  # Return None if no path found

# Example Usage:
maze = generate_maze()
start = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'S'][0]
goal = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'G'][0]

path_bfs = breadth_first_search(maze, start, goal)