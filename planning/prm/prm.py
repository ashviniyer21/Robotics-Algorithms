"""
Python file to run and visualize PRM Planner
Code below assumes Circular obstacles, collision functions can be changed to accomodate for others
Only generates PRM graph, paths can be found using Dijkstra's / A*
"""


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

def get_dist(point1, point2):
    """Computes the distance between 2 points"""
    return np.linalg.norm(point1 - point2)

def get_random_point(bounds):
    """Generates a random point within bounds"""
    x_point = np.random.uniform(bounds[0][0], bounds[0][1])
    y_point = np.random.uniform(bounds[1][0], bounds[1][1])
    return np.array([x_point, y_point])

def get_nearest_neighbor(point, nodes):
    """Finds the nearest node in tree to new point"""
    idx = 0
    min_dist = get_dist(nodes[0], point)
    for i in range(1, len(nodes)):
        curr_dist = get_dist(nodes[i], point)
        if curr_dist < min_dist:
            min_dist = curr_dist
            idx = i
    return (idx, min_dist)

def bound_point_dist(point, neighbor, max_range):
    """Finds point in same direction as generated point within some maximum radius"""
    curr_dist = get_dist(point, neighbor)
    if curr_dist > max_range:
        direction_vec = (point - neighbor) / curr_dist * max_range
        point = neighbor + direction_vec
    return point

def check_in_obstacle(point, obstacles):
    """Checks if a point lies in an obstacle"""
    for obstacle in obstacles:
        x, y, radius = obstacle
        if get_dist(point, np.array([x, y])) < radius:
            return True
    return False

def check_intersection(point1, point2, obstacle):
    """Checks if path between 2 points goes through obstacle"""
    #Obstacle: (c_x, c_y, rad)
    c_x, c_y, r = obstacle
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    a = x1**2 + x2**2 - 2*x1*x2 + y1**2 + y2**2 - 2*y1*y2
    b = 2*x1*x2 - 2*x2*x2 - 2*c_x*x1 + 2*c_x*x2 + 2*y1*y2 - 2*y2*y2 - 2*c_y*y1 + 2*c_y*y2
    c = c_x**2 - 2*c_x*x2 + x2**2 + c_y**2 - 2*c_y*y2 + y2**2 - r**2

    if b**2 - 4*a*c < 0:
        return False
    
    disc = np.sqrt(b**2 - 4*a*c)
    pos_root = (-b + disc) / (2*a)
    neg_root = (-b - disc) / (2*a)
    return (0 <= pos_root <= 1) or (0 <= neg_root <= 1)

def check_intersections(point1, point2, obstacles):
    """Checks if path between 2 points goes through any obstacle"""
    for obstacle in obstacles:
        if check_intersection(point1, point2, obstacle):
            return True
    return False

def find_neighbors(point, nodes, max_dist):
    """Finds all neighbors within a certain radius"""
    neighbors = []
    for i in range(len(nodes)):
        if get_dist(point, nodes[i]) <= max_dist:
            neighbors.append(i)
    return neighbors

def plot(nodes, edges, obstacles, bounds, path=[]):
    """Plots PRM graph"""
    g = nx.Graph()
    for i in range(len(nodes)):
        g.add_node(i)
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            g.add_edge(i, edges[i][j])
    pos = copy.deepcopy(nodes)
    color_map = ["blue"] * len(nodes)

    for p in path:
        color_map[p] = "purple"
    if len(path) > 0:
        color_map[path[0]] = "red"
        color_map[path[-1]] = "green"

    fig, ax = plt.subplots()

    for obstacle in obstacles:
        circle=plt.Circle((obstacle[0],obstacle[1]),obstacle[2])
        ax.add_patch(circle)

    nx.draw(g, pos=pos, node_color = color_map, node_size=10)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    plt.show()

def prm(obstacles, max_range=2, iters=1000, bounds=[[0, 10], [0, 10]]):
    """Runs PRM algorithm"""
    nodes = []
    edges = []
    T_new = -1
    
    # Checking if node is close enough to goal
    for i in range(iters):
        
        #Generating new node & checking if it is a valid node
        random_point = get_random_point(bounds)
        if check_in_obstacle(random_point, obstacles):
            continue
    
        neighbors = find_neighbors(random_point, nodes, max_range)

        # Adding node to graph
        nodes.append(random_point)
        edges.append([])
        T_new += 1

        for neighbor in neighbors:
            if not check_intersections(nodes[neighbor], random_point, obstacles):
                edges[neighbor].append(T_new)
                edges[T_new].append(neighbor)
    return nodes, edges

if __name__ == "__main__":
    """Main function for running PRM"""
    bounds=[[0, 10], [0, 10]]
    #Generating obstacles
    obstacles = []
    for i in range(30):
        center = get_random_point(bounds)
        radius = np.random.uniform(0.1, 1)
        obstacles.append((center[0], center[1], radius))
    
    nodes, edges = prm(obstacles, bounds=bounds)
    plot(nodes, edges, obstacles, bounds)
    #Generating start / end
