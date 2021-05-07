from __future__ import print_function
import os, sys, time, datetime, json, random, math
import numpy as np
import pandas as pd
from statistics import mean
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import tensorflow as tf
import matplotlib.pyplot as plt
import PygameDisplay
import pygame

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


class Qmaze(object):
    def __init__(self, maze, visuals, rat=(0, 0)):
        self._maze = maze
        nrows, ncols = self._maze.shape
        self.visuals = False
        self.exits = 1
        self.walls = -1
        self.paths = 0
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
        self.min_reward = -0.75 * self.maze.size
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
        valid_actions = qmaze.valid_actions()
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


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')

    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    nrows, ncols = qmaze.maze.shape
    count = 0
    while True:
        prev_envstate = envstate
        # get next action
        q = model(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if qmaze.visuals:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            spot = np.where(envstate == rat_mark)
            pyMaze.drawPlayerSprite(math.ceil(float(spot[0][0]) / ncols), spot[0][0] % ncols)
            pyMaze.update()
        if game_status == 'win':
            if qmaze.visuals:
                row, col, mode = qmaze.state
                pyMaze.drawRect(row, col, (0, 183, 255))
                pygame.display.update()
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        print(cell)
        if not qmaze.valid_actions:
            print("Action Fail")
            return False
        if not play_game(model, qmaze, cell):
            print("Playing Fail")
            return False
    return True


class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)

    def predict(self, envstate):
        return self.model(envstate)[0]

    def get_data(self, data_size=10):
        env_size = qmaze.maze.size  # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


def qtrain(model, maze, view, repeat, **opt):
    global global_epoch
    global epsilon
    n_epoch = opt.get('n_epoch', 10000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    prev_accuracy = 0.0
    accuracy_comparison = 0.0
    if repeat:
        weights_file = opt.get('weights_file', "model.h5")  # model.h5
    else:
        weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze, view)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = 50  # history window size
    win_rate = 0.0
    imctr = 1

    log_dir = "logs/spedup"
    writer = tf.summary.create_file_writer(log_dir)
    maze_epoch = 0
    with writer.as_default():
        for epoch in range(global_epoch, global_epoch+n_epoch):
            print("on epoch", epoch)
            loss = 0.0
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            game_over = False

            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()

            n_episodes = 0
            while not game_over:
                valid_actions = qmaze.valid_actions()
                if not valid_actions:
                    break
                prev_envstate = envstate
                # Get next action
                if np.random.rand() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    action = np.argmax(experience.predict(prev_envstate))

                row, col, mode = qmaze.state

                if qmaze.visuals:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            h5file = name + ".h5"
                            json_file = name + ".json"
                            model.save_weights(h5file, overwrite=True)
                            with open(json_file, "w") as outfile:
                                json.dump(model.to_json(), outfile)
                            print("Saved model!")
                            quit()

                    pyMaze.drawMaze()
                    pyMaze.drawPlayerSprite(col, row)
                    pyMaze.update()

                # print(row, col, mode)

                # Apply action, get reward and new envstate
                envstate, reward, game_status = qmaze.act(action)

                if game_status == 'win':
                    if qmaze.visuals:
                        row, col, mode = qmaze.state
                        pyMaze.drawRect(row, col, (0, 183, 255))
                        pyMaze.update()
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                # Store episode (experience)
                episode = [prev_envstate, action, reward, envstate, game_over]

                experience.remember(episode)

                n_episodes += 1
            if global_epoch % 10 == 0 and len(experience.memory) > 0:
                # Train neural network model
                inputs, targets = experience.get_data(data_size=data_size)

                h = model.fit(
                    inputs,
                    targets,
                    epochs=8,
                    batch_size=16,
                    verbose=0,
                )
                print('Trained!')
                tf.summary.scalar("epoch_loss", h.history["loss"][0], step=epoch)
                tf.summary.scalar("episode_reward", qmaze.total_reward, step=epoch)
                tf.summary.scalar("win_rate", sum(win_history) / len(win_history), step=epoch)
                writer.flush()

                loss = model.evaluate(inputs, targets, verbose=0)
                qmaze.loss_memory.append(loss)
                """if len(qmaze.loss_memory) > qmaze.max_loss_memory:
                    qmaze.loss_memory.pop(0)
                """
            elif len(experience.memory) == 0:
                print('No memory')

            global_epoch = global_epoch+1
            win_rate = sum(win_history) / len(win_history)

            if len(win_history) > hsize and sum(win_history) > 0.0 and maze_epoch + 1 % 10 == 0:
                accuracy_comparison = abs(prev_accuracy - win_rate)
                prev_accuracy = win_rate

            maze_epoch = maze_epoch + 1
            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.5f} | " \
                       "time: {} "
            print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))

            # adjust epsilon
            if game_status == 'win' and epsilon >= 0.05:
                epsilon -= .5 / qmaze.maze.size

            experience.memory.clear()
            # check end condition
            if len(win_history) > hsize and accuracy_comparison < .002:
                print("Plateaued at epoch %i" % epoch)
                win_history.clear()
                print(win_history)

                break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    print("Saved model!")
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    qmaze.completing = True
    # fin_check = completion_check(model, qmaze)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    # qmaze.memory.clear()
    return seconds  # fin_check


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


"""
maze = np.array([
    [1., 0., 1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 0., 1., 1.],
    [1., 1., 1., 0., 1., 1., 1.],
    [1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 0., 0.],
    [1., 1., 1., 0., 1., 1., 1.],
])
"""

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
rat_mark = .5  # The current rat cell will be painted by gray 0.5
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
global_epoch = 0


directory = "RandomTrainingMazes/"

reps = False

for file in os.listdir(directory):
    # Exploration factor
    epsilon = 0.1
    filename = os.fsdecode(file)
    mazeFile = os.path.join(directory, filename)
    myFile = pd.read_csv(mazeFile, sep=',', header=None)
    maze = pd.DataFrame(myFile).to_numpy(dtype="float16")

    # Initialize the maze
    qmaze = Qmaze(maze, False)

    if qmaze.visuals:

        dispL = 700
        pyMaze = PygameDisplay.Maze(maze, dispL, qmaze.walls, qmaze.exits)
        pyMaze.drawMaze()
        pyMaze.update()

    # show(qmaze)
    # plt.show()
    model = build_model(maze)

    time = qtrain(model, maze, qmaze.visuals, reps, n_epoch=100000, max_memory=8 * maze.size, data_size=128)

    '''while not finished:
        print("Model was not finished training! Restarting program...")
        qmaze.completing = False
        _time, finished = qtrain(model, maze, qmaze.visuals, epochs=1000, max_memory=8 * maze.size, data_size=64)
        time += _time
    '''
    print("Training successful after %s." % (format_time(time)))
    reps = True
