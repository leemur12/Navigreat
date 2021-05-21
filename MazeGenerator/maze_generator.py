import numpy as np
import pandas as pd
import time
import os
import random
import helper as MGen

NUMBER_OF_MAZES= 8000
FOLDER_NAME= "RandomTrainingMazes"
DIMENSIONS= 9
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
FOLDER= os.path.join(BASE_DIR, FOLDER_NAME)

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)


for i in range(NUMBER_OF_MAZES):
    m = MGen.Maze()
    algo= random.randint(0,4)

    # 0 Backtracking
    # 1 Eller
    # 2 Sidewinder
    # 3 Prim
    # 4 Kruskal
    # 5 Random

    m.create(DIMENSIONS, DIMENSIONS, algo)
    # if algo<5:
    #     m.removeWalls(DIMENSIONS//2)
    m.addExits(1)


    filename= "Maze"+ str(i)+".csv"
    np.savetxt(os.path.join(FOLDER, filename), m.maze.astype(int), delimiter=",", fmt= '%d')













