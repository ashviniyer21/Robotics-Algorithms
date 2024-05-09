import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

def get_dist(point1, point2):
    return np.linalg.norm(point1 - point2)

def get_random_point(bounds):
    x_point = np.random.uniform(bounds[0][0], bounds[0][1])
    y_point = np.random.uniform(bounds[1][0], bounds[1][1])
    return np.array([x_point, y_point])

def get_nearest_neighbor(point, nodes):
    idx = 0
    min_dist = get_dist(nodes[0], point)
    for i in range(1, len(nodes)):
        curr_dist = get_dist(nodes[i], point)
        if curr_dist < min_dist:
            min_dist = curr_dist
            idx = i
    return (idx, min_dist)

def bound_point_dist(point, neighbor, max_range):
    curr_dist = get_dist(point, neighbor)
    if curr_dist > max_range:
        direction_vec = (point - neighbor) / curr_dist * max_range
        point = neighbor + direction_vec
    return point

def check_in_obstacle(point, obstacles):
    for obstacle in obstacles:
        x, y, radius = obstacle
        if get_dist(point, np.array([x, y])) < radius:
            return True
    return False

def check_intersection(point1, point2, obstacle):
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
    for obstacle in obstacles:
        if check_intersection(point1, point2, obstacle):
            return True
    return False

def find_neighbors(point, nodes, max_dist):
    neighbors = []
    for i in range(len(nodes)):
        if get_dist(point, nodes[i]) <= max_dist:
            neighbors.append(i)
    return neighbors

def plot(start, goal, parents, nodes, obstacles, bounds, path=[]):
    g = nx.Graph()
    for i in range(len(parents) + 1):
        g.add_node(i)
    for i in range(1, len(parents)):
        g.add_edge(i, parents[i])
    pos = copy.deepcopy(nodes)
    pos.append(goal)
    color_map = ["blue"] * len(pos)
    color_map[len(nodes) - 1] = "orange"

    for p in path:
        color_map[p] = "purple"
    color_map[0] = "red"
    color_map[len(nodes)] = "green"

    fig, ax = plt.subplots()

    for obstacle in obstacles:
        circle=plt.Circle((obstacle[0],obstacle[1]),obstacle[2])
        ax.add_patch(circle)

    nx.draw(g, pos=pos, node_color = color_map, node_size=100)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    plt.show()

def rrtstar(start, goal, obstacles, max_range=2, threshold=2, bounds=[[0, 10], [0, 10]]):
    nodes = [start]
    costs = [0]
    parents = [-1]
    T_new = 0
    while get_dist(nodes[T_new], goal) > threshold or check_intersections(nodes[T_new], goal, obstacles):
        random_point = get_random_point(bounds)
        neighbor_idx, neighbor_dist = get_nearest_neighbor(random_point, nodes)
        random_point = bound_point_dist(random_point, nodes[neighbor_idx], max_range)
        neighbor_idx, neighbor_dist = get_nearest_neighbor(random_point, nodes)
        if check_intersections(nodes[neighbor_idx], random_point, obstacles):
            continue
    
        neighbors = find_neighbors(random_point, nodes, max_range)

        nodes.append(random_point)
        T_new += 1
        parents.append(neighbor_idx)
        costs.append(neighbor_dist + costs[neighbor_idx])

        for neighbor in neighbors:
            if costs[T_new] + get_dist(random_point, nodes[neighbor]) < costs[neighbor]:
                if not check_intersections(random_point, nodes[neighbor], obstacles):
                    costs[neighbor] = costs[T_new] + get_dist(random_point, nodes[neighbor])
                    parents[neighbor] = T_new
        plot(start, goal, parents, nodes, obstacles, bounds)
    nodes.append(goal)
    parents.append(T_new)
    costs.append(costs[T_new] + get_dist(goal, nodes[T_new]))
    
    path_idx = T_new + 1
    # Recovering nodes to traverse backwards
    path = []
    viz_path = []
    while path_idx != -1:
        path.append(nodes[path_idx])
        viz_path.append(path_idx)
        path_idx = parents[path_idx]
    plot(start, goal, parents, nodes, obstacles, bounds, viz_path)
    return path

if __name__ == "__main__":
    bounds=[[0, 10], [0, 10]]
    obstacles = []
    for i in range(30):
        center = get_random_point(bounds)
        radius = np.random.uniform(0.1, 1)
        obstacles.append((center[0], center[1], radius))
    start = get_random_point(bounds)
    while check_in_obstacle(start, obstacles):
        start = get_random_point(bounds)
    goal = get_random_point(bounds)
    while check_in_obstacle(goal, obstacles):
        goal = get_random_point(bounds)
    print(rrtstar(start, goal, obstacles))