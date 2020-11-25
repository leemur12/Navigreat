import numpy as np
import pygame
from time import time,sleep
from random import randint as r
import random
import pickle
n = 8    # side length of the maze (nxn) DO NOT GO ABOVE 10
scrx = n*100 # set screen width and height
scry = n*100
background = (51,51,51) # set background color
screen = pygame.display.set_mode((scrx,scry)) # initialize the screen with specified dimensions
""" Hard coded mazes
colors = [(51,51,51),(51,51,51),(255,0,0),(51,51,51),(255,0,0),(255,255,0),(51,51,51),(255,0,0),(51,51,51),(255,0,0),(255,255,0),(51,51,51)
# ,(51,51,51),(0,255,0),(51,51,51),(255,255,0)]
# reward = np.array([[-1,-5,-1,-1],[-1,1,-5,10],[-5,-1,1,-1],[-1,-5,-1,1]])
# terminals = [1,6,7,8,13]
# colors = [(51,51,51),(51,51,51),(51,51,51),(51,51,51),(51,51,51),(255,0,0),(255,0,0),(51,51,51),(51,51,51),(51,51,51),(0,255,0),(51,51,51)
# ,(51,51,51),(51,51,51),(51,51,51),(51,51,51)]
# reward = np.array([[0,0,0,0],[0,-1,0,0],[0,-1,1,0],[0,0,0,0]])
# terminals = [5,9,10]
"""
colors = [(51,51,51) for i in range(n**2)] # initialize the maze array with all blank spaces
reward = np.zeros((n,n)) # initialize the reward array with all blank spaces
goals = 1 # number of end goals to have in the maze
terminals = [] # initialize the array of points that end the game (empty)
penalities = 13 # number of walls to have
while penalities != 0: # loop to create the desired number of walls
    i = r(0,n-1) # i and j are random indices in the array
    j = r(0,n-1)
    if reward[i,j] == 0 and [i,j] != [0,0] and [i,j] != [n-1,n-1]: # check that the random spot isn't taken already
        reward[i,j] = -1 # set the corresponding reward array spot to a penalty
        penalities-=1 # one less wall needs to be created
        colors[n*i+j] = (255,0,0) # set the color of the wall square to red
        terminals.append(n*i+j)
reward[n-1,n-1] = 1 # set the bottom right square to be the end
goals-=1 # one less goal needs to be created
colors[n**2 - 1] = (0,255,0) # set the color of the end to be green
terminals.append(n**2 - 1) # set the end square to be a square that restarts the game
while goals != 0: # while more goals need to be created
    i = r(0,n-1) # i and j are random indices in the array
    j = r(0,n-1)
    if reward[i,j] == 0 and [i,j] != [0,0]: # check that the random spot isn't taken already
        reward[i,j] = 1 # set the corresponding reward array spot to be a victory
        goals-=1 # one less goal needs to be created
        colors[n*i+j] = (0,255,0) # set the color of the goal to be green
        terminals.append(n*i+j) # set the goal square to be a square that restarts the game

# f = open("Q.txt","r")
Q = np.zeros((n**2,4))  # create the guiding array, with a column 4 slots per square of the maze
                        # format: [Up]
                        #         [Down]
                        #         [Left]
                        #         [Right]

# Q = pickle.loads(f.read())
# f.close()
actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3} # create a dictionary, connecting the action name with a number
# states = {0 : 0,10 : 1,-1 : 2,1 : 3} #nothing,reward,penalty,goal
states = {} # initialize "states" array (empty)
k = 0 # counting variable
for i in range(n): # iterate through each maze square
    for j in range(n):
        states[(i,j)] = k # assign each square in the states grid the index of the corresponding square in the maze
        k+=1 # increment the counting variable
alpha = 0.01 # initialize the alpha value, learning rate, how much to learn from the future states
gamma = 0.9 # initialize the gamma value
current_pos = [0,0] # initialize the starting position as the top left
epsilon = 0.25 # initialize the chance of taking a random action as 1/4
def layout(): # defining the layout function to draw the base maze
    c = 0 # counting number to track which square to draw
    for i in range(0,scrx,100): # every 100 pixels (grid square) left to right
        for j in range(0,scry,100): # every 100 pixels (grid square) top to bottom
            pygame.draw.rect(screen,(255,255,255),(j,i,j+100,i+100),0) # draw a white exterior rectangle around the grid square
            pygame.draw.rect(screen,colors[c],(j+3,i+3,j+95,i+95),0) # draw a the grid square, color depending on the maze layout
            c+=1 # increment to the next square!
def select_action(current_state): # defining the function to select the next move based on the current position
    global current_pos,epsilon # use the global variables, current_pos and epsilon in this function
    possible_actions = [] # initialize the array of possible actions (empty)
    if np.random.uniform() <= epsilon:  # if the random number from 0 to 1 is less than the "random chance" variable, epsilon
                                        # choose a random possible action
        if current_pos[1] != 0: # if the player is not on the left wall
            possible_actions.append("left") # add left to the list of possible moves
        if current_pos[1] != n-1: # if the player is not on the right wall
            possible_actions.append("right") # add right to the list of possible moves
        if current_pos[0] != 0: # if the player is not on the ceiling
            possible_actions.append("up") # add up to the list of possible moves
        if current_pos[0] != n-1: # if the player is not on the floor
            possible_actions.append("down") # add down to the list of possible moves
        action = actions[possible_actions[r(0,len(possible_actions) - 1)]]  # choose a random move from the possible moves,
                                                                            # then set "action" to the corresponding integer
                                                                            # from the dictionary
    else: # if the random number is greater than the "random chance" variable,
          # choose an action based on the "Q" grid, grading the possible moves.
          # A.K.A. act smart
        m = np.min(Q[current_state]) # set "m" to the lowest value in the current square's column in the "Q" grid
        if current_pos[0] != 0: # if player is not on the ceiling
            possible_actions.append(Q[current_state,0]) # add the Q-value for the "up" option of the current grid to the
                                                        # list of possible actions
        else: # if on the top wall
            possible_actions.append(m - 100) # make up not an option, set its slot to basically -100 in reward
        if current_pos[0] != n-1: # if player is not on the floor
            possible_actions.append(Q[current_state,1]) # add the Q-value for "down" to possible actions
        else: # if on the floor
            possible_actions.append(m - 100) # make down not an option
        if current_pos[1] != 0: # if the player is not on the left wall
            possible_actions.append(Q[current_state,2]) # add the Q-value for "left" to possible actions
        else: # if on the left wall
            possible_actions.append(m - 100) # make left not an option
        if current_pos[1] != n-1: # if the player is not on the right wall
            possible_actions.append(Q[current_state,3]) # add the Q-value for "right" to possible actions
        else: # if on the right wall
            possible_actions.append(m - 100) # make right not an option
        # action = np.argmax(possible_actions)
        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)])
          # ^ if there are multiple actions that have the same Q-value, choose one randomly. otherwise, choose the option
          # with the maximum Q-value
        return action # return the chosen action (int)
def episode(): # define episode function, runs through one "move" in the game
    global current_pos,epsilon # use the global variables, current_pos and epsilon in this function
    current_state = states[(current_pos[0],current_pos[1])] # get the array id of the current grid square
    action = select_action(current_state) # call the function to choose the next action based on the current position
    if action == 0: # if action is up, move 1 row up
        current_pos[0] -= 1
    elif action == 1: # if action is down, move 1 row down
        current_pos[0] += 1
    elif action == 2: # if action is left, move 1 row left
        current_pos[1] -= 1
    elif action == 3: # if action is right, move 1 row right
        current_pos[1] += 1
    new_state = states[(current_pos[0],current_pos[1])] # determine the new grid slot
    if new_state not in terminals: # if the new grid slot is not an "end slot"
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] + gamma*(np.max(Q[new_state])) - Q[current_state,action])
        # increase the Q value of the current slot (not the one moved to) by a fraction of the sum of the reward of the new state
        # and the difference in Q value between the current and next slot.
        # Essentially, if the move made it more likely for the player to reach the goal, the current square is labeled
        # as a better option, because it led to the step that made the player get closer to the goal.
    else: # player moved into an "end slot" and the game needs to be restarted
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] - Q[current_state,action])
        # increase the Q value of the current slot (not the one moved to) by a fraction of the difference the reward
        # of the new slot and the Q value of the current slot. 2 possible "reward" values for a game end, -1 or 1.
        # If the game ends in a win, the current slot gets a large boost in Q value, as it's right next to the exit.
        # if the game is a loss, the current slot takes a significant loss in Q value, as it's next to a wall
        # This makes the player more likely to find the exit, or less likely to hit a wall.
        current_pos = [0,0] # reset the current position to the top left
        epsilon -= 1e-3 # make the player SLIGHTLY less likely to choose a random action. Once this =0, player will always
                        # do the same thing, ideally getting to the end every time.


run = True # variable to check if the window is still open
current_pos = [0,0] #
while run:
    screen.fill(background) # draw the background again
    layout() # draw the maze
    pygame.draw.circle(screen,(25,129,230),(current_pos[1]*100 + 50,current_pos[0]*100 + 50),30,0) # draw the player at
                                                                                                # the current position
    for event in pygame.event.get(): # when something happens while the window is open
        if (event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
        # if the thing that happens is the X button clicked or escape key is pressed
            run = False # stop running through the loop
    pygame.display.flip() # update the display
    episode() # run through one move

pygame.quit() # exit the program
print(epsilon) # print out the "learning progress" (chance to do a random move)
# f = open("Q.txt","w")
# f.write(pickle.dumps(Q))
# f.close()