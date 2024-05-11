"""Python file to run and visualize icp algorithm"""

import numpy as np
from perception.helper.kd_tree import KDTree
import matplotlib.pyplot as plt

def generate_random_points(num, low, high):
    """Generates random pointcloud with specificed number of points and minimum sizes"""
    return np.random.uniform(low, high, (num, len(low)))

"""Uses closest point correspondence w/ kd-tree"""
def find_correspondences(points1, points2):
    kd_tree = KDTree(points2, 5)
    correspondences = []
    error = 0
    for i in range(points1.shape[0]):
        point = points1[i]
        matching_point, dist = kd_tree.find_closest_point(point)
        error += dist * dist
        correspondences.append((point, matching_point))
    error = np.sqrt(error)
    error /= points1.shape[0]
    return correspondences, error

"""Finds transformation using SVD approach. Uses 2d for example, can be 3d"""
def compute_transform(correspondences):

    x0 = np.zeros(2)
    y0 = np.zeros(2)
    H = np.zeros((2, 2))
    for correspondence in correspondences:
       x0 += correspondence[0]
       y0 += correspondence[1]
    x0 /= len(correspondences)
    y0 /= len(correspondences)
    for correspondence in correspondences:
       x, y = correspondence
       x = np.copy(x)
       y = np.copy(y)
       x -= x0
       y -= y0
       H += np.array([[y[0] * x[0], y[0] * x[1]], [y[1] * x[0], y[1] * x[1]]])
    U, s, V = np.linalg.svd(H, full_matrices=False)
    R = V.T @ U.T
    t = y0 - R @ x0
    return R, t

def transform_points(rotation, translation, points):
    for i in range(points.shape[0]):
        points[i] = rotation @ points[i] + translation
    return points

def icp(points1, points2, threshold=0.01):
    """Runs icp algorithm to find optimal rotation / translation to align points2 with points1"""
    error = np.inf
    while error > threshold:
        correspondences, new_error = find_correspondences(points1, points2)
        if new_error < threshold or new_error > error:
             break
        error = new_error
        rotation, translation = compute_transform(correspondences)
        points1 = transform_points(rotation, translation, points1)
    
    return points1

if __name__ == "__main__":
    points1 = generate_random_points(10, [0, 0], [10, 10])
    points2 = generate_random_points(10, [5, 5], [15, 15])
    plt.scatter(points1[:,0], points1[:,1], c='blue')
    plt.scatter(points2[:,0], points2[:,1], c='red')
    points1 = icp(points1, points2)
    plt.scatter(points1[:,0], points1[:,1], c='green')
    plt.show()
