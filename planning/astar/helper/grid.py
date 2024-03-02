"""Class to provide grid structure for graph based planners"""

import numpy as np

class Grid:
    """Class to provide grid structure for graph based planners"""

    def __init__(self, length, width, obstacle_percentage=0.25):
        self.grid = np.random.choice([0, 1], size=(length, width),
                                     p=[1 - obstacle_percentage, obstacle_percentage])
        self.rows = length
        self.columns = width


    def is_obstacle(self, row, column):
        """Returns if a point on the grid has an obstacle (1) or not (0).
        Returns -1 if not a valid grid point"""

        if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
            return -1
        return self.grid[row, column]

    def find_random_free_space(self):
        """Generates a random space from the grid that is not occupied"""

        row = -1
        column = -1
        while self.is_obstacle(row, column) != 0:
            row = np.random.randint(self.rows)
            column = np.random.randint(self.columns)
        return (row, column)
