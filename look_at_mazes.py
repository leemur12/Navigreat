import display
import os
import pygame
import pandas as pd
import qmaze as q



def getNewMaze(file_index):
    filename = maze_list[file_index]
    mazeFile = os.path.join(directory, filename)
    myFile = pd.read_csv(mazeFile, sep=',', header=None)
    maze = pd.DataFrame(myFile).to_numpy(dtype="float16")
    return maze

directory = "RandomTrainingMazes/"

dispL=700

WALLS=-1
EXITS=1

isLooking=True

maze_list= os.listdir(directory)
file_index=0
maze= getNewMaze(file_index)

pyMaze = display.Maze(maze, dispL, WALLS, EXITS)
pyMaze.drawMaze()
pygame.display.update()


while isLooking:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        if event.type== pygame.KEYDOWN:
            if event.key== pygame.K_LEFT:


                if file_index>0:
                    file_index-=1
                    maze=getNewMaze(file_index)
                    pyMaze.updateMaze(maze)
                    pyMaze.drawMaze()
                    pygame.display.update()

        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_RIGHT:

                if file_index<len(maze_list):
                    file_index+=1
                    maze = getNewMaze(file_index)

                    pyMaze.updateMaze(maze)
                    pyMaze.drawMaze()
                    pygame.display.update()
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_DOWN:
                maze = getNewMaze(file_index)
                print(maze)
                print()
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_UP:
                qmaze= q.Qmaze(maze)
                print(qmaze.free_cells)
                print()


