from __future__ import print_function
import os, sys, time, datetime, json, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
import display
import pygame
import qmaze as q
import model

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


def qtrain(maze_model, maze, view, repeat, data_size=128,  **opt):


    global global_epoch
    global epsilon

    global_epoch=0
    n_epoch=60000
    prev_accuracy = 0.0
    accuracy_comparison = 0.0

    start_time = datetime.datetime.now()


    # Initialize experience replay object

    hsize = 50  # history window size
    long_win_history=[]


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
        short_win_history = []  # history of win/lose game
        loss=0

        maze_epoch = 0
        with writer.as_default():
            for epoch in range(global_epoch, global_epoch+n_epoch):
                maze_epoch+=1

                print("on epoch", epoch)
                loss = 0.0
                rat_cell = random.choice(qmaze.free_cells)
                qmaze.reset(rat_cell)
                game_over = False

                # get initial envstate (1d flattened canvas)
                envstate = qmaze.observe4()

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
                        action = np.argmax(maze_model.predict(prev_envstate))

                    row, col, mode = qmaze.state

                    if qmaze.visuals:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                maze_model.save()
                                quit()
                        pyMaze.update(row, col)


                    # Apply action, get reward and new envstate
                    envstate, reward, game_status = qmaze.act(action)


                    if game_status == 'win':
                        if qmaze.visuals:
                            row, col, mode = qmaze.state
                            pyMaze.drawRect(row, col, (0, 183, 255))
                            pyMaze.update(row, col)

                        long_win_history.append(1)
                        short_win_history.append(1)
                        game_over = True
                    elif game_status == 'lose' or n_episodes>25:
                        long_win_history.append(0)
                        short_win_history.append(0)

                        game_over = True
                    else:
                        game_over = False

                    # Store episode (experience)
                    episode = [prev_envstate, action, reward, envstate, game_over]

                    maze_model.remember(episode)

                    # Train neural network model
                    if len(maze_model.memory) > 10:
                        h = maze_model.train(data_size)
                        tf.summary.scalar("epoch_loss", h.history["loss"][0], step=epoch)
                        tf.summary.scalar("ep_reward", reward, step=epoch)
                        tf.summary.scalar("game_reward", qmaze.total_reward, step=epoch)
                        writer.flush()
                        loss = h.history['loss'][0]


                    n_episodes += 1

                if len(maze_model.memory) > 10:
                    tf.summary.scalar("short_win_rate", sum(short_win_history) / len(short_win_history), step=epoch)
                    tf.summary.scalar("long_win_rate", sum(long_win_history) / len(long_win_history), step=epoch)
                    tf.summary.scalar("number_moves", n_episodes, step=epoch)
                    writer.flush()


                    #model.evaluate(inputs, targets, verbose=0)
                    qmaze.loss_memory.append(loss)


                win_rate = sum(short_win_history) / len(short_win_history)

                if len(short_win_history) > hsize and sum(short_win_history) > 0.0 and maze_epoch + 1 % 10 == 0:
                    accuracy_comparison = abs(prev_accuracy - win_rate)
                    prev_accuracy = win_rate


                dt = datetime.datetime.now() - start_time
                t = format_time(dt.total_seconds())
                template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.5f} | " \
                           "time: {} "
                print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(long_win_history), win_rate, t))

                # adjust epsilon
                if game_status == 'win' and epsilon >= 0.05:
                    epsilon -= .1 / qmaze.maze.size


                # check end condition
                if len(short_win_history) > hsize and accuracy_comparison < .002:
                    print("Plateaued at epoch %i" % epoch)
                    short_win_history.clear()


                    break
            maze_model.memory.clear()

            global_epoch += maze_epoch

        # Save trained model weights and architecture, this will be used by the visualization code
        maze_model.save()




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

name="4inputs"
timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir="logs/"+name+timestr
num_actions = len(actions_dict)
global_epoch = 0
# Exploration factor
epsilon = 0.1
num_actions=4

directory = "RandomTrainingMazes/"

reps = False

maze=np.zeros((9,9))
qmaze = q.Qmaze(maze, False)

if qmaze.visuals:

    dispL=700
    pyMaze = display.Maze(maze, dispL, qmaze.walls, qmaze.exits)
    pyMaze.drawMaze()
    pygame.display.update()


input_shape= maze.size*4
maze_model= model.MazeModel(input_shape, num_actions, name)
qtrain(maze_model, maze, qmaze.visuals, reps, epochs=1000, max_memory=8 * maze.size, data_size=24)