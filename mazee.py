import random
from collections import deque
import heapq
import math
import time

expanded_nodes_bfs = []
expanded_nodes_dfs = []
expanded_nodes_A = []


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

# # Example usage:
# random_maze = generate_maze()
# for row in random_maze:
#     print(' '.join(row))


def depth_first_search(maze, start, goal):
    stack = [(start, [start])]
    visited = set()
    #expanded_nodes_dfs=[]
    while stack:
        (current_i, current_j), path_dfs = stack.pop()
       
        if (current_i, current_j) == goal:
                    # print("_________________DFS______________________")
                    # print("Path Length DFS:", len(path_dfs) - 1)  # Subtract 1 to exclude the starting point
                    # print("Path Nodes DFS:", path_dfs)
                    # print("Number of Nodes Expanded DFS:", len(expanded_nodes_dfs))
                    # print("Expanded Nodes DFS:", expanded_nodes_dfs)
                    return path_dfs  # Return the path if goal reached
        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))
            expanded_nodes_dfs.append((current_i, current_j))

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
                    stack.append(((neighbor_i, neighbor_j), path_dfs + [(neighbor_i, neighbor_j)]))

    return None  # Return None if no path found

def breadth_first_search(maze, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    #expanded_nodes_bfs = []
    while queue:
        (current_i, current_j), path_bfs = queue.popleft()
        path_bfs1=path_bfs
        if (current_i, current_j) == goal:
                    # print("_________________BFS______________________")
                    # print("Path Length BFS:", len(path_bfs) - 1)  # Subtract 1 to exclude the starting point
                    # print("Path Nodes BFS:", path_bfs)
                    # print("Number of Nodes Expanded BFS:", len(expanded_nodes_bfs ))
                    # print("Expanded Nodes BFS:", expanded_nodes_bfs )
                    return path_bfs  # Return the path if goal reached

        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))
            expanded_nodes_bfs .append((current_i, current_j))

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
                    queue.append(((neighbor_i, neighbor_j), path_bfs + [(neighbor_i, neighbor_j)]))

    return None  # Return None if no path found

def heuristic(current, goal):
    # Euclidean distance heuristic
    return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

def astar(maze, start, goal):
    priority_queue = [(0, start, [start])]
    visited = set()
    #expanded_nodes_A=[]
    while priority_queue:
        _, (current_i, current_j), path_A = heapq.heappop(priority_queue)
        path_A1=path_A
        if (current_i, current_j) == goal:
                            # print("_________________A*______________________")
                            # print("Path Length A*:", len(path_A) - 1)  # Subtract 1 to exclude the starting point
                            # print("Path Nodes A*:", path_A)
                            # print("Number of Nodes Expanded A*:", len(expanded_nodes_A))
                            # print("Expanded Nodes A*:", expanded_nodes_A)
                            return path_A  # Return the path if goal reached
        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))
            expanded_nodes_A.append((current_i, current_j))

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
                    new_path = path_A + [(neighbor_i, neighbor_j)]
                    priority = len(new_path) + heuristic((neighbor_i, neighbor_j), goal)
                    heapq.heappush(priority_queue, (priority, (neighbor_i, neighbor_j), new_path))

    return None  # Return None if no path found


def visualize_path1(maze, path):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if (i, j) == path[0]:
                print('\033[91mS\033[0m', end=' ')  # Red color for start
            elif (i, j) == path[-1]:
                print('\033[92mG\033[0m', end=' ')  # Green color for goal
            elif (i, j) in path:
                print('\033[94m.\033[0m', end=' ')  # Blue color for path
            else:
                print(maze[i][j], end=' ')
        print()

def evaluate_algorithm(algorithm, maze, start, goal):
    start_time = time.time()
    path = algorithm(maze, start, goal)
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        # 'path_length': len(path) if path else float('inf'),
        # 'nodes_expanded': len(set(node for step in path for node in step)) if path else float('inf'),
        'execution_time': execution_time
    }

def display_results(path,expendednodes):
  
    print("Path Length :", len(path) - 1)  # Subtract 1 to exclude the starting point
    print("Path Nodes :", path)
    print("Number of Nodes Expanded :", len(expendednodes))
    print("Expanded Nodes :", expendednodes)

# Example usage:
maze = generate_maze()

print("Generated Maze")
for row in maze:
    print(' '.join(row))



start = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'S'][0]
goal = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'G'][0]

start_time = time.time()
found_path = depth_first_search(maze, start, goal)
end_time = time.time()
execution_time = end_time - start_time


if found_path:
    print("_________________DFS______________________")
    display_results(found_path ,expanded_nodes_dfs)
    visualize_path1(maze, found_path)
    print("Execution Time:",execution_time)
else:
    print("No path found.")

start_time = time.time()
found_path_bfs = breadth_first_search(maze, start, goal)
end_time = time.time()
execution_time = end_time - start_time

if found_path_bfs:
    print("_________________BFS______________________")
    display_results(found_path_bfs,expanded_nodes_bfs)
    visualize_path1(maze, found_path_bfs)
    print("Execution Time:",execution_time)
else:
    print("No path found.")

start_time = time.time()
found_path_astar = astar(maze, start, goal)
end_time = time.time()
execution_time = end_time - start_time

if found_path_astar:
    print("_________________A*______________________")
    display_results(found_path_astar,expanded_nodes_A)
    visualize_path1(maze, found_path_astar)
    print("Execution Time:",execution_time)
else:
    print("No path found.")

import pygame
import sys

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up Pygame
pygame.init()

# Constants
CELL_SIZE = 30
FPS = 10

# Initialize font globally

font = None

def initialize_font():
    global font
    font = pygame.font.Font(None, 36)

def draw_maze(screen, maze):
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            color = WHITE if cell == '.' else BLACK
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def draw_path(screen, path, color):
    for i, j in path:
        pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))


def draw_start_and_goal(screen, start, goal):
    i, j = start
    pygame.draw.rect(screen, RED, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    text_surface = font.render('S', True, WHITE, RED)
    screen.blit(text_surface, (j * CELL_SIZE + CELL_SIZE // 3, i * CELL_SIZE + CELL_SIZE // 3))

    i, j = goal
    pygame.draw.rect(screen, GREEN, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    text_surface = font.render('G', True, WHITE, GREEN)
    screen.blit(text_surface, (j * CELL_SIZE + CELL_SIZE // 3, i * CELL_SIZE + CELL_SIZE // 3))

def visualize_algorithm(screen, algorithm, maze, start, goal, color_explored, color_solution,explored_path,solution_path):
    clock = pygame.time.Clock()


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_maze(screen, maze)

        if explored_path:
            draw_path(screen, explored_path, color_explored)

        if solution_path:
            draw_path(screen, solution_path, color_solution)
            
        draw_start_and_goal(screen, start, goal)
        pygame.display.flip()
        clock.tick(FPS)

    # pygame.quit()
    # sys.exit()

start = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'S'][0]
goal = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'G'][0]

initialize_font()

screen_dfs = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("DFS Visualization")
visualize_algorithm(screen_dfs, depth_first_search, maze, start, goal, '#808080', BLUE,expanded_nodes_dfs,found_path)

screen_bfs = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("BFS Visualization")
visualize_algorithm(screen_bfs, breadth_first_search, maze, start, goal, '#808080', BLUE,expanded_nodes_bfs,found_path_bfs)


screen_astar = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("A* Visualization")
visualize_algorithm(screen_astar, astar, maze, start, goal, '#808080', BLUE,expanded_nodes_A,found_path_astar)
