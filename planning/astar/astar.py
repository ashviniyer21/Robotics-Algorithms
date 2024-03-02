"""Python file to run and visualize A* algorithm"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from helper.grid import Grid

def visualize_astar(start, goal, grid, prev_nodes, nodes, highlights=None):
    """Visualizes A* algorithm at a given step"""
    array = np.copy(grid.grid)
    for node in nodes:
        array[node[0], node[1]] = 4
    for node in prev_nodes.keys():
        if node not in nodes:
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

def astar(start, goal, grid, heuristic):
    """Performs A* search algorithm given a start and goal"""

    open_set = set()
    open_set.add(start)

    g_values = {}
    g_values[start] = 0

    f_values = {}
    f_values[start] = heuristic(start, goal)

    prev_nodes = {}
    prev_nodes[start] = None

    nodes = set()
    nodes.add(start)

    while len(nodes) > 0:
        curr_node = None
        curr_f_value = None
        curr_g_value = None
        for node in nodes:
            temp_f_value = f_values[node]
            if curr_node is None or curr_f_value > temp_f_value:
                curr_f_value = temp_f_value
                curr_g_value = g_values[node]
                curr_node = node

        nodes.remove(curr_node)

        if curr_node == goal:
            final_path = []
            temp_node = goal
            while temp_node is not None:
                final_path.append(temp_node)
                temp_node = prev_nodes[temp_node]
            final_path.reverse()
            visualize_astar(start, goal, grid, prev_nodes, nodes, highlights=final_path)
            return final_path

        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (curr_node[0] + i, curr_node[1] + j)
                if (i != 0 or j != 0) and grid.is_obstacle(neighbor[0], neighbor[1]) == 0:
                    dist = np.sqrt(i**2 + j**2)
                    g_value = curr_g_value + dist
                    if neighbor not in g_values or g_values[neighbor] > g_value:
                        prev_nodes[neighbor] = curr_node
                        g_values[neighbor] = g_value
                        f_values[neighbor] = g_value + heuristic(neighbor, goal)
                        if neighbor not in nodes:
                            nodes.add(neighbor)
        visualize_astar(start, goal, grid, prev_nodes, nodes, highlights=[curr_node])

    return []

def manhattan_heuristic(node, goal):
    """Heuristic for diagonal distance"""

    dx = abs(node[0] - goal[0])
    dy = abs(node[1] - goal[1])
    return dx + dy

if __name__ == '__main__':
    grid = Grid(20, 20)
    start = grid.find_random_free_space()
    goal = grid.find_random_free_space()
    astar(start, goal, grid, manhattan_heuristic)
