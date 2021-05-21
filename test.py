
import os
import display
import qmaze as q
import model
import tensorflow as tf
import time
import pandas as pd
import random
from keras.models import load_model
import pygame


def nn_test(visuals=False):
    total_wins = 0
    with writer.as_default():
        for epoch in range(len(file_list)):

            file_name= os.path.join(test_directory, file_list[epoch])
            myFile = pd.read_csv(file_name, sep=',', header=None)
            maze = pd.DataFrame(myFile).to_numpy()

            pyMaze = display.Maze(maze, dispL, -1, 1)

            qmaze= q.Qmaze(maze)

            start= random.choice(qmaze.free_cells)
            qmaze.reset(start)

            game_over=False

            envstate=qmaze.observe3()

            total_episodes=0
            print(epoch)
            while not game_over:
                prev_envstate= envstate

                val_actions= qmaze.valid_actions()
                if not val_actions:
                    print("trapped")

                calcs= model.predict(prev_envstate)

                action= val_actions[0]
                reward= calcs[action]


                if len(val_actions)>1:
                    for act in val_actions[1:]:
                        if calcs[act]>reward:
                            action=act
                            reward=calcs[act]



                row, col, mode = qmaze.state


                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()

                if visuals:
                    pyMaze.update(row, col)
                    time.sleep(0.05)

                # Apply action, get reward and new envstate
                envstate, reward, game_status = qmaze.act(action)
                print("newMove")
                print(envstate)
                print(reward)
                if game_status == 'win':

                    row, col, mode = qmaze.state
                    pyMaze.drawRect(row, col, (0, 183, 255))
                    pyMaze.update(row, col)

                    total_wins+=1
                    game_over = True


                if total_episodes>20:
                    game_over=True

                total_episodes+=1

            tf.summary.scalar("moves_per_epoch", total_episodes, step=epoch)
            tf.summary.scalar("win_rate", total_wins/(epoch+1), step=epoch)
            tf.summary.scalar("wins", total_wins, step=epoch)
            writer.flush()
    print(total_wins/1000)
def baseline_test():

    total_wins = 0
    with writer.as_default():
        for epoch in range(len(file_list)):

            file_name = os.path.join(test_directory, file_list[epoch])
            myFile = pd.read_csv(file_name, sep=',', header=None)
            maze = pd.DataFrame(myFile).to_numpy()

            pyMaze = display.Maze(maze, dispL, -1, 1)

            qmaze = q.Qmaze(maze)

            start = random.choice(qmaze.free_cells)
            qmaze.reset(start)
            game_over = False

            #envstate = qmaze.observe2()

            total_episodes = 0
            while not game_over:
                prev_envstate = envstate

                val_actions = qmaze.valid_actions()
                if not val_actions:
                    print("trapped")



                action = random.choice(val_actions)

                row, col, mode = qmaze.state

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()
                    pyMaze.update(row, col)

                # Apply action, get reward and new envstate
                envstate, reward, game_status = qmaze.act(action)

                if game_status == 'win':
                    row, col, mode = qmaze.state
                    pyMaze.drawRect(row, col, (0, 183, 255))
                    pyMaze.update(row, col)

                    total_wins += 1
                    game_over = True

                if total_episodes > 20:
                    game_over = True

                total_episodes += 1

            tf.summary.scalar("moves_per_epoch", total_episodes, step=epoch)
            tf.summary.scalar("win_rate", total_wins / (epoch + 1), step=epoch)
            tf.summary.scalar("wins", total_wins, step=epoch)
            writer.flush()
    print(total_wins/1000)



test_directory= "RandomTestMazes/"
name="new_agent_tracker"

timestr = time.strftime("%Y%m%d-%H%M%S")
test_log_directory= "Results/"+name+timestr
dispL=700

input_shape=81*3
num_actions=4


model= model.MazeModel(input_shape, num_actions, name)
model.load_weights()



writer = tf.summary.create_file_writer(test_log_directory)
file_list= os.listdir(test_directory)

nn_test(True)
