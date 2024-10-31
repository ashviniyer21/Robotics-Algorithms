"""Python file to run and visualize A* algorithm"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from planning.helper.grid import Grid

def visualize_pfield(start, goal, grid, prev_nodes, new_obs, highlights=None):
    """Visualizes A* algorithm at a given step"""
    array = np.copy(grid.grid)
    for node in new_obs:
        array[node[0], node[1]] = 4
    for node in prev_nodes:
        array[node[0], node[1]] = 5

    for highlight in highlights:
        array[highlight[0], highlight[1]] = 6

    array[start[0], start[1]] = 2
    array[goal[0], goal[1]] = 3

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    cmap = colors.ListedColormap(['white', 'black', 'green', 'red', 'yellow', 'blue', 'orange'])
    norm = colors.BoundaryNorm(bounds, cmap.N)

    _, ax = plt.subplots()
    ax.imshow(array, cmap=cmap, norm=norm)

    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, grid.rows, 1))
    ax.set_yticks(np.arange(-.5, grid.columns, 1))
    plt.show()

def diagonal_heuristic(node, goal):
    """Heuristic for diagonal distance"""
    D = 1
    D2 = np.sqrt(2)
    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def min_wall_dist(grid, node):
    return (min(node[0], grid.rows - node[0]), min(node[1], grid.columns - node[1]))

def calc_potential_field(grid, goal):
    goal_w = 1
    wall_w = 1
    scores = np.zeros((grid.rows, grid.columns))
    obstacles = []
    for i in range(grid.rows):
        for j in range(grid.columns):
            if grid.is_obstacle(i, j):
                scores[i, j] = np.inf
                obstacles.append((i, j))
            else:
                x_wall, y_wall = min_wall_dist(grid, (i, j))
                scores[i, j] = goal_w * diagonal_heuristic([i, j], goal) + wall_w / (1 + x_wall) + wall_w / (1 + y_wall)
    for obstacle in obstacles:
        add_obstacle(scores, obstacle)
    return scores

def add_obstacle(scores, obstacle):
    obs_w = 1
    for i in range(grid.rows):
        for j in range(grid.columns):
            if not grid.is_obstacle(i, j):
                scores[i, j] += obs_w / diagonal_heuristic([i, j], obstacle)
    scores[obstacle[0], obstacle[1]] = np.inf

def potential_field(grid, scores, pos):
    best_neighbor = (-1, -1)
    best_score = np.inf
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbor = (pos[0] + i, pos[1] + j)
            if (i != 0 or j != 0) and grid.is_obstacle(neighbor[0], neighbor[1]) == 0:
                if scores[neighbor[0], neighbor[1]] < best_score:
                    best_score = scores[neighbor[0], neighbor[1]]
                    best_neighbor = neighbor
    return best_neighbor

if __name__ == '__main__':
    grid = Grid(50, 50, obstacle_percentage=0.25)
    start = grid.find_random_free_space()
    goal = grid.find_random_free_space()
    pos = start
    prev_nodes = set()
    fake_obs = []
    scores = calc_potential_field(grid, goal)
    visualize_pfield(start, goal, grid, prev_nodes, fake_obs, highlights=[pos])
    while diagonal_heuristic(pos, goal) > 0:
        if pos in prev_nodes:
            fake_obs.append(pos)
            add_obstacle(scores, pos)
            prev_nodes = set()
        else:
            prev_nodes.add(pos)
        pos = potential_field(grid, scores, pos)

        visualize_pfield(start, goal, grid, prev_nodes, fake_obs, highlights=[pos])
