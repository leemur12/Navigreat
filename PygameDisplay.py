import numpy as np
import pygame
import math


class Maze:
    def __init__(self, maze, game_display, disp_len, wall=-1, win=5):
        self.maze = maze
        self.gameDisplay = game_display
        self.mazeR = len(maze)
        self.mazeC = len(maze[0])
        self.wall = wall
        self.win = win

        self.blockLen = disp_len / max(self.mazeR, self.mazeC)

    def update(self, maze):
        self.maze = maze

    def draw(self, islines=False):
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

    def drawQTableValuesArrow(self, qTable):

        arrowImg = pygame.image.load('arrow.png')
        arrowImg = pygame.transform.scale(arrowImg, (int(self.blockLen * .8), int(self.blockLen * .8)))

        for i in range(qTable.shape[0]):
            actions = qTable[i]
            if not np.sum(actions) == 0:

                yvec = actions[0] - actions[1]
                xvec = actions[3] - actions[2]

                if xvec == 0:
                    if yvec > 0:
                        deg = 90
                    else:
                        deg = 270

                elif xvec < 0:
                    deg = math.degrees(math.atan(yvec / xvec)) + 180
                else:
                    deg = math.degrees(math.atan(yvec / xvec))

                y = self.blockLen * math.floor(i / self.mazeR)
                x = self.blockLen * (i % self.mazeR)

                self.blitRotateCenter(self.gameDisplay, arrowImg, (x, y), deg)

    def drawQtableValuesNum(self, qTable):
        myFont = pygame.font.SysFont("Times New Roman", 18)

        for i in range(qTable.shape[0]):
            maxValue = np.amax(qTable[i])

            y = self.blockLen * math.floor(i / self.mazeR)
            x = self.blockLen * (i % self.mazeR)

            label = myFont.render(str(maxValue)[0:5], 1, (0, 0, 0))
            self.gameDisplay.blit(label, (x, y))

    def blitRotateCenter(self, surf, image, topleft, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

        surf.blit(rotated_image, new_rect.topleft)

    def MapValues(self, x, input_start, input_end, output_start, output_end):
        mapVal = int(x * (output_end - output_start) / (input_end - input_start))

        if mapVal < output_start:
            return output_start
        elif mapVal > output_end:
            return output_end
        else:
            return mapVal


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
