import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import math
import time

class Maze:
    def __init__(self, maze, disp_len=700, wall=-1, win=5):

        self.maze = maze
        pygame.init()

        rows, cols = maze.shape
        square_size = int(disp_len / max(rows, cols))
        self.gameDisplay = pygame.display.set_mode(size=(square_size * cols, square_size * rows))
        pygame.display.set_caption('Maze')
        self.gameDisplay.fill((255, 255, 255))
        self.mazeR = len(maze)
        self.mazeC = len(maze[0])
        self.wall = wall
        self.win = win

        self.blockLen = disp_len / max(self.mazeR, self.mazeC)

        self.PlayerSprite= PlayerSprite(self.blockLen, self.gameDisplay)

    def update(self, row, col):
        self.drawMaze()
        self.drawPlayerSprite(col, row)
        pygame.display.update()

    def updateMaze(self, maze):
        self.maze = maze

    def drawMaze(self, islines=False):
        self.gameDisplay.fill((255, 255, 255))
        ypos = 0
        for i in self.maze:

            xpos = 0
            for j in i:

                if j == self.wall:

                    pygame.draw.rect(self.gameDisplay, (55, 55, 55), [xpos, ypos, self.blockLen + 1, self.blockLen + 1])
                elif j == self.win:
                    pygame.draw.rect(self.gameDisplay, (0, 255, 0), [xpos, ypos, self.blockLen + 1, self.blockLen + 1])

                xpos += self.blockLen

            ypos += self.blockLen

        if islines:
            lineThick = self.MapValues(self.blockLen, 1, 80, 1, 4)
            yLine = self.blockLen
            for i in range(0, self.mazeC):
                pygame.draw.line(self.gameDisplay, (0, 0, 0), (yLine, 0), (yLine, self.mazeR * self.blockLen),
                                 lineThick)
                yLine += self.blockLen

            xLine = self.blockLen
            for i in range(0, self.mazeR):
                pygame.draw.line(self.gameDisplay, (0, 0, 0), (0, xLine), (self.blockLen * self.mazeC, xLine),
                                 lineThick)
                xLine += self.blockLen

    def drawRect(self, row, col, color=(255, 255, 255)):
        pygame.draw.rect(self.gameDisplay, color, [col * self.blockLen, row * self.blockLen, self.blockLen + 1, self.blockLen + 1])

    def drawPlayerSprite(self, x, y):
        self.PlayerSprite.draw(x,y)


    def drawPath(self, path, isReversed=False):
        if isReversed:
            p=reversed(path)
        for cell in p:
            pygame.draw.rect(self.gameDisplay, (0, 0, 255), [cell[1]*self.blockLen, cell[0]*self.blockLen, self.blockLen + 1, self.blockLen + 1])
            pygame.display.update()
            time.sleep(0.03)


class PlayerSprite:
    def __init__(self, blockL, gameDisplay, value=.2):
        try:
            spri = pygame.image.load("sprite.png")
            self.sprite = pygame.transform.scale(spri, (int(blockL * .8), int(blockL * .8)))
            self.preloadedSprite = True
        except:
            self.preloadedSprite = False
        self.stride = blockL
        self.gameDisplay = gameDisplay
        self.value = value

    def draw(self, x, y):
        if self.preloadedSprite:
            self.gameDisplay.blit(self.sprite, (x * self.stride, y * self.stride))
        else:
            pygame.draw.circle(self.gameDisplay, (0, 0, 0),
                               (int(x * self.stride + self.stride / 2), int(y * self.stride + self.stride / 2)),
                               int(self.stride / 3))
