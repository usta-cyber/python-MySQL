import random
from collections import deque
import heapq
import math
import time
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




def depth_first_search(maze, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        (current_i, current_j), path = stack.pop()

        if (current_i, current_j) == goal:
            return path  

        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))

          
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
                    stack.append(((neighbor_i, neighbor_j), path + [(neighbor_i, neighbor_j)]))

    return None  
def breadth_first_search(maze, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        (current_i, current_j), path = queue.popleft()

        if (current_i, current_j) == goal:
            return path  

        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))


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

    return None  

def heuristic(current, goal):

    return math.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

def astar(maze, start, goal):
    priority_queue = [(0, start, [start])]
    visited = set()

    while priority_queue:
        _, (current_i, current_j), path = heapq.heappop(priority_queue)

        if (current_i, current_j) == goal:
            return path  

        if (current_i, current_j) not in visited:
            visited.add((current_i, current_j))

           
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
                    new_path = path + [(neighbor_i, neighbor_j)]
                    priority = len(new_path) + heuristic((neighbor_i, neighbor_j), goal)
                    heapq.heappush(priority_queue, (priority, (neighbor_i, neighbor_j), new_path))

    return None  


def evaluate_algorithm(algorithm, maze, start, goal):
    start_time = time.time()
    path = algorithm(maze, start, goal)
    end_time = time.time()
    execution_time = end_time - start_time

    return {
        'path_length': len(path) if path else float('inf'),
        'nodes_expanded': len(set(node for step in path for node in step)) if path else float('inf'),
        'execution_time': execution_time
    }



maze = generate_maze()

print("Generated Maze")
for row in maze:
    print(' '.join(row))



start = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'S'][0]
goal = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'G'][0]

found_path = depth_first_search(maze, start, goal)

if found_path:
    print("Path found using DFS:",found_path)

else:
    print("No path found.")

found_path_bfs = breadth_first_search(maze, start, goal)

if found_path_bfs:
    print("Path found using BFS:",found_path_bfs)
else:
    print("No path found.")


found_path_astar = astar(maze, start, goal)

if found_path_astar:
    print("Path found using A* algorithm:",found_path_astar)
else:
    print("No path found.")


dfs_result = evaluate_algorithm(depth_first_search, maze, start, goal)


bfs_result = evaluate_algorithm(breadth_first_search, maze, start, goal)


astar_result = evaluate_algorithm(astar, maze, start, goal)

# Print results
print("DFS Results:")
print(dfs_result)

print("\nBFS Results:")
print(bfs_result)

print("\nA* Results:")
print(astar_result)




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

def visualize_algorithm(screen, algorithm, maze, start, goal, color_explored, color_solution):
    clock = pygame.time.Clock()

    explored_path = []
    solution_path = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)
        draw_maze(screen, maze)


        if not explored_path:
            explored_path = algorithm(maze, start, goal)

        if explored_path:
            draw_path(screen, explored_path, color_explored)

            if not solution_path:
                solution_path = explored_path

            if solution_path:
                draw_path(screen, solution_path, color_solution)
                solution_path = []
        draw_start_and_goal(screen, start, goal)
        pygame.display.flip()
        clock.tick(FPS)

    # pygame.quit()
    # sys.exit()

maze = generate_maze()
start = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'S'][0]
goal = [(i, j) for i, row in enumerate(maze) for j, val in enumerate(row) if val == 'G'][0]

initialize_font()

screen_bfs = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("BFS Visualization")
visualize_algorithm(screen_bfs, breadth_first_search, maze, start, goal, RED, BLUE)


screen_dfs = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("DFS Visualization")
visualize_algorithm(screen_dfs, depth_first_search, maze, start, goal, RED, BLUE)


screen_astar = pygame.display.set_mode((len(maze[0]) * CELL_SIZE, len(maze) * CELL_SIZE))
pygame.display.set_caption("A* Visualization")
visualize_algorithm(screen_astar, astar, maze, start, goal, RED, BLUE)
