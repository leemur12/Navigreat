import numpy as np
import pandas as pd
import time
import os
import MazeGeneratorHelper as MGen

NUMBER_OF_MAZES= 10
FOLDER_NAME= "RandomTrainingMazes"
DIMENSIONS= 18
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
FOLDER= os.path.join(BASE_DIR, FOLDER_NAME)

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

for i in range(NUMBER_OF_MAZES):
    m = MGen.Maze()
    m.create(DIMENSIONS, DIMENSIONS, 5) #5 means that it uses the "spawn random block" algorithm.
                                        #See MazeGeneratorHelper to see all the different algorithms.
    m.addExits()

    filename= "Maze"+ str(i)+".csv"
    np.savetxt(os.path.join(FOLDER, filename), m.maze.astype(int), delimiter=",", fmt= '%d')













