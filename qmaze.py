import random
import numpy as np

rat_mark = 0.8

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

class Qmaze(object):
    def __init__(self, maze, visuals, rat=(0, 0)):
        self._maze = maze
        nrows, ncols = self._maze.shape
        self.visuals = True
        self.exits = 1.0
        self.walls = -1.0
        self.paths = 0.0
        self.target_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == self.exits]
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == self.paths]
        rat = random.choice(self.free_cells)
        if rat not in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)
        self.last_visited = rat
        self.completing = False
        self.loss_memory = list()
        self.max_loss_memory = 50

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.1 * self.maze.size
        self.total_reward = 0
        self.visited = set()
        self.last_visited = rat

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > self.walls and (rat_row, rat_col) not in self.visited:
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
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if (rat_row, rat_col) in self.target_cells:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        valid_actions = self.valid_actions()
        if len(valid_actions) == 1:
            return -0.75
        """if (rat_row, rat_col) in self.visited:
            if (rat_row, rat_col) == self.last_visited:
                print("Whoops")
                return -2.0
            return -0.5"""
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04
        if mode == 'start':
            return 0.0

    def act(self, action):
        self.last_visited = self.rat
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] == 0.5:
                    canvas[r, c] = 0.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        envstate = canvas.reshape((1, -1))
        return envstate

    def game_status(self):
        if self.total_reward < self.min_reward:
            # print(self.total_reward)
            return 'lose'
        rat_row, rat_col, mode = self.state
        if (rat_row, rat_col) in self.target_cells:
            # print(self.total_reward)
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows - 1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols - 1:
            actions.remove(2)

        if row > 0 and self.maze[row - 1, col] == self.walls:
            actions.remove(1)
        if row < nrows - 1 and self.maze[row + 1, col] == self.walls:
            actions.remove(3)

        if col > 0 and self.maze[row, col - 1] == self.walls:
            actions.remove(0)
        if col < ncols - 1 and self.maze[row, col + 1] == self.walls:
            actions.remove(2)

        if row > 0 and (row - 1, col) in self.visited and self.completing:
            actions.remove(1)
        if col > 0 and (row, col - 1) in self.visited and self.completing:
            actions.remove(0)
        if row < nrows - 1 and (row + 1, col) in self.visited and self.completing:
            actions.remove(3)
        if col < ncols - 1 and (row, col + 1) in self.visited and self.completing:
            actions.remove(2)

        return actions

