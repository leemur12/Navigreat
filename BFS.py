import numpy as np
import pandas as pd
import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import PygameDisplay
import pygame

from collections import deque

directory = "TrainingMazes/"

file= os.listdir(directory)[0]
filename = os.fsdecode(file)

pdMaze = pd.read_csv(os.path.join(directory, filename), sep=',', header=None)
maze = pd.DataFrame(pdMaze).to_numpy(dtype="float16")

matrix= np.zeros(maze.shape)

start= (1,1)

matrix[start]=1

next_cells= deque([start])
win_cells=[]

step=2
isWin=False

start_time = time.time()
while next_cells:

    q_len= len(next_cells)
    for i in range(q_len):
        check_cell= next_cells.popleft()
        r, c= check_cell

        if maze[r,c]==1:
            win_cells.append((r,c))
            continue

        if r>0 and maze[r-1, c] !=-1 and matrix[r-1,c]==0:
            next_cells.append((r-1,c))
            matrix[r-1, c] =step

        if r<maze.shape[0] and maze[r+1,c]!=-1 and matrix[r+1,c]==0:
            next_cells.append((r+1, c))
            matrix[r + 1, c] = step

        if c>0 and maze[r, c-1]!=-1 and matrix[r, c-1]==0:
            next_cells.append((r, c-1))
            matrix[r, c-1] = step

        if c<maze.shape[1] and maze[r, c+1]!=-1 and matrix[r,c+1]==0:
            next_cells.append((r, c+1))
            matrix[r, c+1] = step

    step=step+1

def find_path(endpoint):
    r, c= endpoint

    path=[(r,c)]
    step =matrix[r,c]
    while step>1:
        if r>0 and matrix[r-1,c]== step-1:
            r-=1
            path.append((r,c))
            step-=1
        if r<maze.shape[0]-1 and matrix[r+1,c]== step-1:
            r+=1
            path.append((r, c))
            step-=1

        if c>0 and matrix[r, c-1]== step-1:
           c-=1
           path.append((r, c))
           step-=1

        if c<maze.shape[1]-1 and matrix[r,c+1]== step-1:
            c+=1
            path.append((r, c))
            step-=1
    path.pop()
    return path

print(time.time()-start_time)



dispL = 800
rows, cols = maze.shape
square_size = int(dispL / max(rows, cols))
pygame.init()
gameDisplay = pygame.display.set_mode(size=(square_size * cols, square_size * rows))
pygame.display.set_caption('Maze')
gameDisplay.fill((255, 255, 255))
pyMaze = PygameDisplay.Maze(maze, gameDisplay, dispL, -1, 1)


for win_cell in win_cells:
    path= find_path(win_cell)
    pyMaze.draw()
    for cell in reversed(path):
        r,c=cell

        pyMaze.drawRect(r,c, (255,0,0))
        pygame.display.update()
        time.sleep(0.05)

    pyMaze.draw()
    pygame.display.update()




