import os, sys, time, datetime, json, random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU

import PygameDisplay
import pygame


class Qmaze(object):
    def __init__(self, maze, rat=(0, 0)):
        self._maze = maze
        nrows, ncols = self._maze.shape

        self.target_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 2.0]
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        rat = random.choice(self.free_cells)
        if rat not in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)

        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()
        print(self.maze)

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if (rat_row, rat_col) in self.target_cells:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            valid_actions = self.valid_actions()
            if len(valid_actions) == 1:
                return -0.75
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04
        if mode == 'start':
            return 0.0

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))

        return envstate

    def draw_env(self):
        canvas = np.copy(self._maze)
        # print(self._maze)

        nrows, ncols = self.maze.shape
        # clear all visual marks

        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat

        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if (rat_row, rat_col) in self.target_cells:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self._maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self._maze[row - 1, col] == 0.0:
            actions.remove(1)
        if row < nrows - 1 and self._maze[row + 1, col] == 0.0:
            actions.remove(3)

        if col > 0 and self._maze[row, col - 1] == 0.0:
            actions.remove(0)
        if col < ncols - 1 and self._maze[row, col + 1] == 0.0:
            actions.remove(2)

        return actions


def run_model(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "model.h5")  # model.h5
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)
    for cell in qmaze.free_cells:
        qmaze.reset(cell)
        envstate = qmaze.observe()
        game_over = False
        count = 0
        while not game_over:

            count = count + 1
            prev_envstate = envstate
            # get next action
            q = model(prev_envstate).numpy()[0]
            action = np.argmax(q)

            while action not in qmaze.valid_actions():
                q[action] += -999
                action = np.argmax(q)

            # apply action, get rewards and new state
            envstate, reward, game_status = qmaze.act(action)

            row, col, mode = qmaze.state
            # print(qmaze.valid_actions())
            # print(row, col, mode)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            pyMaze.drawMaze()
            pyMaze.drawPlayerSprite(col, row)
            pygame.display.update()
            if game_status == 'win':
                game_over = True
                # print("won")
            elif game_status == 'lose' or count >= 20:
                game_over = True
                # print("lost")
            else:
                game_over = False
            time.sleep(0.1)


def build_model(maze, lr=0.001):
    model_build = Sequential()
    model_build.add(Dense(maze.size, input_shape=(maze.size,)))
    model_build.add(PReLU())
    model_build.add(Dense(maze.size))
    model_build.add(PReLU())
    model_build.add(Dense(num_actions))
    model_build.compile(optimizer='adam', loss='mse')
    return model_build


myFile = pd.read_csv('TrainingMazes/NGArray.csv', sep=',', header=None)
maze = pd.DataFrame(myFile).to_numpy()

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = 0.5  # The current rat cell will be painted by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Initialize the maze

dispL = 700
pyMaze = PygameDisplay.Maze(maze,  dispL, -1, 1)
pyMaze.drawMaze()
pygame.display.update()

model = build_model(maze)
run_model(model, maze)
