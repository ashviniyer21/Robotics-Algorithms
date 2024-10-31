import pygame
import sys
from quadtree import QuadTreeMap, CellState
from objects import Point, Circle
import numpy as np
from astar import astar

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
ORANGE = (200, 100, 0)
BLUE = (0, 0, 200)
PURPLE = (200, 0, 200)
GREY = (100, 100, 100)
COLORS = [WHITE, BLACK, GREY, GREEN, RED]
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 800
BORDER_COLOR = (255, 255, 255)


def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(WHITE)
    start = Point(np.random.uniform(0, WINDOW_WIDTH), np.random.uniform(0, WINDOW_HEIGHT))
    goal = Point(np.random.uniform(0, WINDOW_WIDTH), np.random.uniform(0, WINDOW_HEIGHT))
    quadtree = QuadTreeMap(10, 80, WINDOW_WIDTH, WINDOW_HEIGHT)
    circles = []
    for i in range(10):
        x, y = np.random.uniform(0, WINDOW_WIDTH), np.random.uniform(0, WINDOW_HEIGHT)
        rad = np.random.uniform(10, 50)
        circles.append((x, y, rad))
        quadtree.insert(Circle(x, y, rad), CellState.OCCUPIED)
    path = astar(start, goal, quadtree)
    highlights = []
    length = 0
    # highlights = path[1:length]
    should_resid = False
    grid_view = False
    show_neighbors = False
    while True:
        # highlights = [(ORANGE, path[1:length])]
        highlights = [(ORANGE, [path[length]])]
        if should_resid and length > 0:
            highlights.append((PURPLE, path[0:length]))
        if show_neighbors:
            highlights.append((BLUE, path[length].get_neighbors()))
        if grid_view:
            drawQuadTree(quadtree, highlights)
        else:
            for (x, y, rad) in circles:
                pygame.draw.circle(SCREEN, BLACK, (x, y), rad)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    grid_view = not grid_view
                if event.key == pygame.K_LEFT:
                    length -= 1
                if event.key == pygame.K_RIGHT:
                    length += 1
                if event.key == pygame.K_a:
                    should_resid = not should_resid
                if event.key == pygame.K_b:
                    show_neighbors = not show_neighbors
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()


def draw_square(quadnode, color):
    rect = pygame.Rect(quadnode.left_x, quadnode.bottom_y, quadnode.size, quadnode.size)
    pygame.draw.rect(SCREEN, color, rect)
    pygame.draw.rect(SCREEN, BORDER_COLOR, rect, 1)
def drawQuadTree(quadtree, highlights=[]):
    for i in range(len(quadtree.grid)):
        for j in range(len(quadtree.grid[i])):
            squares = quadtree.grid[i][j].get_squares()
            # print(len(squares))
            for square in squares:
                draw_square(square, COLORS[square.state.value])
    
    for color, highlight_list in highlights:
        for highlight in highlight_list:
            draw_square(highlight, color)

    # print("STOP")


if __name__ == "__main__":
    main()