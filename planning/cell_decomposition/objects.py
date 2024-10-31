from abc import ABC, abstractmethod

class Objects(ABC):
    @abstractmethod
    def collides(self, object):
        pass

class Point(Objects):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def collides(self, object):
        if(type(object) == AARectangle):
            return object.collides(self)
        elif (type(object) == Point):
            return self.x == object.x and self.y == object.y
        else:
            print("Cannot check collision between", type(self), "and", type(object))
        return False

class AARectangle(Objects):
    def __init__(self, left_x, bottom_y, width, height):
        self.x1 = left_x
        self.y1 = bottom_y
        self.x2 = left_x + width
        self.y2 = bottom_y + height

    def collides(self, object):
        if type(object) == AARectangle:
            return self._aarect_check(object)
        elif type(object) == Point:
            return self._point_check(object)
        elif type(object) == Circle:
            return object.collides(self)
        else:
            print("Cannot check collision between", type(self), "and", type(object))
        return False
    
    def _aarect_check(self, object):
        return (self.x1 < object.x2 and self.x2 > object.x1 and self.y1 < object.y2 and self.y2 > object.y1) 

    def _point_check(self, object):
        return self.x1 <= object.x and self.x2 > object.x and self.y1 <= object.y and self.y2 > object.y
    
class Circle(Objects):
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.r = radius

    def collides(self, object):
        if type(object) == Point:
            return (object.x - self.x) ** 2 + (object.y - self.y) ** 2 < self.r ** 2
        elif type(object == AARectangle):
            xn = max(object.x1, min(self.x, object.x2))
            yn = max(object.y1, min(self.y, object.y2))
            return (xn - self.x) ** 2 + (yn - self.y) ** 2 < self.r ** 2
        else:
            print("Cannot check collision between", type(self), "and", type(object))
        return False