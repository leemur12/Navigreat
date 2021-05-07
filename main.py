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
import qmaze as q
import experience as exp
import build_model as bm

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


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
    # Initialize experience replay object

    experience = exp.Experience(model, max_memory=max_memory)

    hsize = 50  # history window size
    win_rate = 0.0
    imctr = 1

    writer = tf.summary.create_file_writer(log_dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        mazeFile = os.path.join(directory, filename)
        myFile = pd.read_csv(mazeFile, sep=',', header=None)
        maze = pd.DataFrame(myFile).to_numpy(dtype="float16")
        pyMaze.updateMaze(maze)

        # Initialize the maze
    # Construct environment/game from numpy array: maze (see above)
        qmaze = q.Qmaze(maze, view)
        win_history = []  # history of win/lose game

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
                                h5file = os.path.join(model_dir,name + ".h5")
                                json_file = os.path.join(model_dir, name + ".json")
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

                # Train neural network model
                if experience.memory:
                    inputs, targets = experience.get_data(data_size=data_size)

                    h = model.fit(
                        inputs,
                        targets,
                        epochs=8,
                        batch_size=16,
                        verbose=0,
                    )

                    tf.summary.scalar("epoch_loss", h.history["loss"][0], step=epoch)
                    tf.summary.scalar("episode_reward", qmaze.total_reward, step=epoch)
                    tf.summary.scalar("win_rate", sum(win_history) / len(win_history), step=epoch)
                    writer.flush()

                    loss = model.evaluate(inputs, targets, verbose=0)
                    qmaze.loss_memory.append(loss)

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
                    epsilon -= .1 / qmaze.maze.size


                # check end condition
                if len(win_history) > hsize and accuracy_comparison < .002:
                    print("Plateaued at epoch %i" % epoch)
                    win_history.clear()
                    print(win_history)

                    break
                experience.memory.clear()

        # Save trained model weights and architecture, this will be used by the visualization code
        h5file = os.path.join(model_dir, name + ".h5")
        json_file = os.path.join(model_dir, name + ".json")
        model.save_weights(h5file, overwrite=True)
        with open(json_file, "w") as outfile:
            json.dump(model.to_json(), outfile)
        print("Saved model!")
        print('files: %s, %s' % (h5file, json_file))
        print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
        print("Checking for completion. This will take a while.")
        qmaze.completing = True
        # fin_check = completion_check(model, qmaze)
        end_time = datetime.datetime.now()
        dt = datetime.datetime.now() - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)



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

model_dir="models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir="logs/"+ timestr
num_actions = len(actions_dict)
global_epoch = 0
# Exploration factor
epsilon = 0.1


directory = "RandomTrainingMazes/"

reps = False

maze=np.zeros((17,17))
qmaze = q.Qmaze(maze, False)

if qmaze.visuals:

    dispL=700
    pyMaze = PygameDisplay.Maze(maze, dispL, qmaze.walls, qmaze.exits)
    pyMaze.drawMaze()
    pyMaze.update()

model = bm.build_model(maze, num_actions)

qtrain(model, maze, qmaze.visuals, reps, epochs=1000, max_memory=8 * maze.size, data_size=128)



