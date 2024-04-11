"""File contains implementation for KD Data Structure"""

import numpy as np

class KDTree:
    """Class to create KD Tree"""
    def __init__(self, points, max_level, level=0):
        """Initializes KD Tree. Assumes root if max_level / level not specified"""
        if level >= max_level: #If already through all dimensions, look through remaining list of points
            self.points = points
            self.leaf = True
            return
        
        self.leaf = False
        
        self.level = level
        self.max_level = max_level
        self.dim = points.shape[1]

        idx = level % self.dim

        
        self.average = np.mean(points[:,idx])

        small_list = points[points[:, idx] <= self.average, :]
        large_list = points[points[:, idx] > self.average, :]
        
        self.small_node = KDTree(small_list, max_level, level + 1)
        self.large_node = KDTree(large_list, max_level, level + 1)

    
    def find_closest_point(self, point):
        point = np.copy(point)
        distance = np.inf
        best_point = None
        if self.leaf:
            for comp_point in self.points:
                curr_dist = self.compute_distance(comp_point, point)
                if curr_dist < distance:
                    distance = curr_dist
                    best_point = comp_point
            return best_point, distance

        value = point[self.level % self.dim]
        if value < self.average:
            best_point, distance = self.small_node.find_closest_point(point)
            if distance > self.average - value:
                temp_point, temp_dist = self.large_node.find_closest_point(point)
                if temp_dist < distance:
                    best_point = temp_point
                    distance = temp_dist
        else:
            best_point, distance = self.large_node.find_closest_point(point)
            if distance > value - self.average:
                temp_point, temp_dist = self.small_node.find_closest_point(point)
                if temp_dist < distance:
                    best_point = temp_point
                    distance = temp_dist
        
        return best_point, distance

    def compute_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)