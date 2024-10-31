from quadtree import CellState
import heapq
import numpy as np
import time

def distance(node1, node2):
    center1 = (node1.left_x + node1.size / 2, node1.bottom_y + node1.size / 2)
    center2 = (node2.left_x + node2.size / 2, node2.bottom_y + node2.size / 2)
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def heuristic(node, goal):
    center1 = (node.left_x + node.size / 2, node.bottom_y + node.size / 2)
    center2 = (goal.left_x + goal.size / 2, goal.bottom_y + goal.size / 2)
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    
def astar(start, goal, quadtree):
    if quadtree.get_node_at_loc(start).state != CellState.FREE:
        print("Invalid Start Location at", start)
        return None
    if quadtree.get_node_at_loc(goal).state != CellState.FREE:
        print("Invalid Goal Location at", goal)
        return None
    start_node = quadtree.insert(start, CellState.START)[0]
    goal_node = quadtree.insert(goal, CellState.GOAL)[0]
    heap = []
    closed_set = set()
    camefrom = {}
    g_scores = {}
    camefrom[start_node] = None
    g_scores[start_node] = 0
    heapq.heappush(heap, (heuristic(start_node, goal_node), time.time(), start_node))

    while len(heap) > 0:
        _, _, curr_node = heapq.heappop(heap)
        if curr_node in closed_set:
            continue
        if curr_node == goal_node:
            path = []
            while curr_node is not None:
                path.append(curr_node)  # Append the center of the current cell
                curr_node = camefrom[curr_node]
            return path[::-1]  # Reverse the path to get it from start to goal
        for neighbor in curr_node.get_neighbors():
            if neighbor.state == CellState.OCCUPIED:
                continue
            new_g = g_scores[curr_node] + distance(curr_node, neighbor)
            if not neighbor in g_scores or new_g < g_scores[neighbor]:
                camefrom[neighbor] = curr_node
                g_scores[neighbor] = new_g
                heapq.heappush(heap, (new_g + heuristic(neighbor, goal_node), time.time(), neighbor))
        closed_set.add(curr_node)
    print("Failed to Find Path")
    return None