import numpy as np
from enum import Enum
from objects import AARectangle, Point

class CellState(Enum):
    FREE=0
    OCCUPIED=1
    MIXED=2
    START=3
    GOAL=4

class QuadNode:
    def __init__(self, root, left_x, bottom_y, size, minsize, maxsize):
        self.left_x = left_x
        self.bottom_y = bottom_y
        self.state = CellState.FREE
        self.size = size
        self.minsize = minsize
        self.maxsize = maxsize
        self.root = root
        self.children = None
        self.neighbors = {}
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                self.neighbors[(i, j)] = set()
        self.bounding_box = AARectangle(left_x, bottom_y, size, size)

    def create_children(self):
        halfsize = self.size // 2
        self.children = [] # Order is Bottom Left, Bottom Right, Top Left, Top Right
        self.state = CellState.MIXED
        for i in range(2):
            for j in range(2):
                self.children.append(QuadNode(self.root, self.left_x + j * halfsize, self.bottom_y + i * halfsize, halfsize, self.minsize, self.maxsize))
        
        # Corner Update
        for a in range(-1, 2, 2):
            for b in range(-1, 2, 2):
                cidx0 = max(0, b)
                cidx1 = max(0, a)
                neighbors = list(self.neighbors[(a, b)])
                for i in range(len(neighbors)):
                    self.children[2 * cidx0 + cidx1].neighbors[(a, b)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, -b)].add(self.children[2 * cidx0 + cidx1])
                    neighbors[i].neighbors[(-a, -b)].discard(self)

        # Left / Right Update
        for a in range(-1, 2, 2):
            neighbors = list(self.neighbors[(a, 0)])
            for i in range(len(neighbors)):
                cidx = max(0, a)
                bottom_y = neighbors[i].bottom_y
                top_y = neighbors[i].bottom_y + neighbors[i].size
                divider = self.bottom_y + self.size / 2
                if bottom_y < divider:
                    self.children[cidx].neighbors[(a, 0)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, 0)].add(self.children[cidx])
                elif bottom_y == divider:
                    self.children[cidx].neighbors[(a, 1)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, -1)].add(self.children[cidx])
                cidx += 2
                if top_y > divider:
                    self.children[cidx].neighbors[(a, 0)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, 0)].add(self.children[cidx])
                elif top_y == divider:
                    self.children[cidx].neighbors[(a, -1)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, 1)].add(self.children[cidx])
                neighbors[i].neighbors[(-a, 0)].discard(self)
        
        # Bottom / Top Update
        for a in range(-1, 2, 2):
            neighbors = list(self.neighbors[(0, a)])
            for i in range(len(neighbors)):
                cidx = 2 * max(0, a)
                left_x = neighbors[i].left_x
                right_x = neighbors[i].left_x + neighbors[i].size
                divider = self.left_x + self.size / 2
                if left_x < divider:
                    self.children[cidx].neighbors[(0, a)].add(neighbors[i])
                    neighbors[i].neighbors[(0, -a)].add(self.children[cidx])
                elif left_x == divider:
                    self.children[cidx].neighbors[(1, a)].add(neighbors[i])
                    neighbors[i].neighbors[(-1, -a)].add(self.children[cidx])
                cidx += 1
                if right_x > divider:
                    self.children[cidx].neighbors[(0, a)].add(neighbors[i])
                    neighbors[i].neighbors[(0,-a)].add(self.children[cidx])
                elif right_x == divider:
                    self.children[cidx].neighbors[(-1, a)].add(neighbors[i])
                    neighbors[i].neighbors[(1, -a)].add(self.children[cidx])
                neighbors[i].neighbors[(0, -a)].discard(self)
        
        # Interal Node Update
        # for i in range(2):
        #     for j in range(2):
        #         for a in range(-1, 2):
        #             for b in range(-1, 2):
        #                 if (a == 0 and b == 0) or i + a < 0 or j + b < 0 or i + a > 1 or j + b > 1:
        #                     continue
        #                 self.children[2 * i + j].neighbors[(a, b)].add(self.children[2 * (i + a) + (j + b)])
        self.children[0].neighbors[(1, 0)].add(self.children[1])
        self.children[0].neighbors[(0, 1)].add(self.children[2])
        self.children[0].neighbors[(1, 1)].add(self.children[3])
        
        self.children[1].neighbors[(-1, 0)].add(self.children[0])
        self.children[1].neighbors[(0, 1)].add(self.children[3])
        self.children[1].neighbors[(-1, 1)].add(self.children[2])
        
        self.children[2].neighbors[(1, 0)].add(self.children[3])
        self.children[2].neighbors[(0, -1)].add(self.children[0])
        self.children[2].neighbors[(1, -1)].add(self.children[1])
        
        self.children[3].neighbors[(-1, 0)].add(self.children[2])
        self.children[3].neighbors[(0, -1)].add(self.children[1])
        self.children[3].neighbors[(-1, -1)].add(self.children[0])
        
    def merge_children(self):
        # Corner Update
        for a in range(-1, 2, 2):
            for b in range(-1, 2, 2):
                cidx0 = max(0, b)
                cidx1 = max(0, a)
                neighbors = list(self.children[2 * cidx0 + cidx1].neighbors[(a, b)])
                for i in range(len(neighbors)):
                    self.neighbors[(a, b)].add(neighbors[i])
                    neighbors[i].neighbors[(-a, -b)].discard(self.children[2 * cidx0 + cidx1])
                    neighbors[i].neighbors[(-a, -b)].add(self)
        
        # Left / Right Side
        for a in range(-1, 1, 2):
            cidx = max(0, a)
            neighbors = list(self.children[cidx].neighbors[(a, 0)].union(self.children[cidx + 2].neighbors[(a, 0)]))
            for i in range(len(neighbors)):
                self.neighbors[(a, 0)].add(neighbors[i])
                neighbors[i].neighbors[(-a, 0)].discard(self.children[cidx])
                neighbors[i].neighbors[(-a, 0)].discard(self.children[cidx + 2])
                neighbors[i].neighbors[(-a, 0)].add(self)
        
        # Bottom / Top Side
        for a in range(-1, 1, 2):
            cidx = 2 * max(0, a)
            neighbors = list(self.children[cidx].neighbors[(0, a)].union(self.children[cidx + 1].neighbors[(0, a)]))
            for i in range(len(neighbors)):
                self.neighbors[(0, a)].add(neighbors[i])
                neighbors[i].neighbors[(0, -a)].discard(self.children[cidx])
                neighbors[i].neighbors[(0, -a)].discard(self.children[cidx + 1])
                neighbors[i].neighbors[(0, -a)].add(self)
        
        self.state = self.children[0].state
        self.children = None

    def insert(self, object, state):
        if not self.bounding_box.collides(object):
            return []
        if self.size == self.minsize:
            self.state = state
            return [self]
        ret_children = []
        if self.children == None:
            self.create_children()
        for child in self.children:
            ret_children.extend(child.insert(object, state))
        
        if self.children[0].state != CellState.MIXED:
            check_state = self.children[0].state
            merge = True
            for i in range(1, len(self.children)):
                if self.children[i].state != check_state:
                    merge = False
                    break
            if merge:
                self.merge_children()
                return [self]
        return ret_children
    
    def find_node(self, point, target_size, dir):
        if(self.children == None):
            if self.state == CellState.OCCUPIED:
                return []
            return [self]
        if(self.size > target_size):
            for child in self.children:
                if child.bounding_box.collides(point):
                    return child.find_node(point, target_size, dir)
            print("Error in finding Node")
            return None
        else:
            nodes = []
            newdir = dir
            if self.size == target_size:
                newdir = (dir[0] * -1, dir[1] * -1)
            if dir[0] < 1 and dir[1] < 1: #Go bottom left
                nodes.extend(self.children[0].find_node(point, target_size, newdir))
            if dir[0] > -1 and dir[1] < 1: #Go bottom right
                nodes.extend(self.children[1].find_node(point, target_size, newdir))
            if dir[0] < 1 and dir[1] > -1: #Go top left
                nodes.extend(self.children[2].find_node(point, target_size, newdir))
            if dir[0] > -1 and dir[1] > -1: #Go top right
                nodes.extend(self.children[3].find_node(point, target_size, newdir))
            return nodes

    def get_neighbors(self):
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor = self.neighbors[(i, j)]
                if neighbor != None:
                    neighbors.extend(neighbor)
        return neighbors
    
    def get_squares(self):
        if self.children == None:
            return [self]
        ret = []
        for child in self.children:
            ret.extend(child.get_squares())
        return ret
    
    def get_node_at_loc(self, point):
        if not self.bounding_box.collides(point):
            print("None")
            return None
        if self.state == CellState.MIXED:
            idx = -1
            if point.x - self.left_x < self.size / 2:
                if point.y - self.bottom_y < self.size < 2:
                    idx = 0
                else:
                    idx = 2
            else:
                if point.y - self.bottom_y < self.size < 2:
                    idx = 1
                else:
                    idx = 3
            return self.children[idx].get_node_at_loc(point)
        else:
            return self

    def __eq__(self, other):
        if type(other) == QuadNode:
            return self.left_x == other.left_x and self.bottom_y == other.bottom_y and self.size == other.size
        return False
    
    def __hash__(self):
        return hash((self.left_x, self.bottom_y, self.size))

class QuadTreeMap:
    def __init__(self, minsize, maxsize, width, length):
        self.rows = int(np.ceil(length / maxsize))
        self.columns = int(np.ceil(width / maxsize))
        self.size = maxsize
        self.width = width
        self.length = length
        self.grid = []
        self.maxsize = maxsize
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                row.append(QuadNode(self, j * maxsize, i * maxsize, maxsize, minsize, maxsize))
            self.grid.append(row)
        
        for i in range(self.rows):
            for j in range(self.columns):
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if (a == 0 and b == 0) or i + a < 0 or j + b < 0 or i + a >= self.rows or j + b >= self.columns:
                            continue
                        self.grid[i][j].neighbors[(b, a)].add(self.grid[i + a][j + b])


    def find_node(self, target_x, target_y, target_size, dir):
        row = int(target_y // self.maxsize)
        column = int(target_x // self.maxsize)
        if(row < 0 or column < 0 or row >= self.rows or column >= self.columns):
            return None
        return self.grid[row][column].find_node(Point(target_x, target_y), target_size, dir)
    
    def insert(self, object, state):
        ret = []
        for i in range(self.rows):
            for j in range(self.columns):
                ret.extend(self.grid[i][j].insert(object, state))
        return ret

    def get_node_at_loc(self, point):
        target_x = point.x
        target_y = point.y
        row = int(target_y // self.maxsize)
        column = int(target_x // self.maxsize)
        if(row < 0 or column < 0 or row >= self.rows or column >= self.columns):
            return None
        return self.grid[row][column].get_node_at_loc(Point(target_x, target_y))