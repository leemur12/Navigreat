import numpy as np
import pandas as pd
import time
import os
import random
import MazeGeneratorHelper as MGen

NUMBER_OF_MAZES= 5000
FOLDER_NAME= "RandomTrainingMazes"
DIMENSIONS= 18
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
FOLDER= os.path.join(BASE_DIR, FOLDER_NAME)

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)


for i in range(NUMBER_OF_MAZES):
    m = MGen.Maze()
    algo= random.randint(0,5)

    # 0 Backtracking
    # 1 Eller
    # 2 Sidewinder
    # 3 Prim
    # 4 Kruskal
    # 5 Random

    m.create(DIMENSIONS, DIMENSIONS, algo)

    m.addExits()

    filename= "Maze"+ str(i)+".csv"
    np.savetxt(os.path.join(FOLDER, filename), m.maze.astype(int), delimiter=",", fmt= '%d')













