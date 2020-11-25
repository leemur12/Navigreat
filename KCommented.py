import numpy as np
import pygame
from time import time,sleep
from random import randint as r
import random
import pickle


#Creates the pygame display
n = 7
scrx = n*100
scry = n*100
background = (51,51,51)
screen = pygame.display.set_mode((scrx,scry))

#initalizes color array for the entire gameboard
colors = [(51,51,51) for i in range(n**2)]

#initializes rewards board with zeros
reward = np.zeros((n,n))

# goals = 1
terminals = []

#number of wall blocks
penalities = 10

#adds all wall blocks
while penalities != 0:
    #finds a random position on gameboard
    i = r(0,n-1)
    j = r(0,n-1)

    #if the spot is empty and it isn't the first or last positions add the penalty
    if reward[i,j] == 0 and [i,j] != [0,0] and [i,j] != [n-1,n-1]:
        #setting reward value
        reward[i,j] = -1
        penalities-=1

        #the arrays below are flattened to a shape of 49
        #setting color value
        colors[n*i+j] = (255,0,0)

        #set the terminal position
        terminals.append(n*i+j)

#finish block
reward[n-1,n-1] = 1
colors[n**2 - 1] = (0,255,0)
terminals.append(n**2 - 1)

# creating multiple endpoints
# while goals != 0:
#     i = r(0,3)
#     j = r(0,3)
#     if reward[i,j] == 0 and [i,j] != [0,0]:
#         reward[i,j] = 1
#         goals-=1
#         colors[n*i+j] = (0,255,0)
#         terminals.append(n*i+j)

# initializes Q table 49 states, 4 actions
Q = np.zeros((n**2,4))

#pickle loads an already created Q-Table
# Q = pickle.loads(f.read())
# f.close()

#mapping between action name and qtable column
actions = {"up": 0,"down" : 1,"left" : 2,"right" : 3}
# states = {0 : 0,10 : 1,-1 : 2,1 : 3} #nothing,reward,penality,goal

#creates a dictionary of states acceseed with the tuple key of their position
#maps between 7x7 and 49 array
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i,j)] = k
        k+=1

#learning curve variable
alpha = 0.01

#considers future rewards at 90% the worth of immediate rewards
gamma = 0.9
current_pos = [0,0]
epsilon = 0.25

#draws screen
def layout():
    c = 0
    for i in range(0,scrx,100):
        for j in range(0,scry,100):
            pygame.draw.rect(screen,(255,255,255),(j,i,j+100,i+100),0)
            pygame.draw.rect(screen,colors[c],(j+3,i+3,j+95,i+95),0)
            c+=1
def select_action(current_state):
    global current_pos,epsilon
    possible_actions = []

    #choosing whether to move randomly or follow qTable
    if np.random.uniform() <= epsilon:
        #random
        #if possible add the move to the list of possible moves
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n-1:
            possible_actions.append("right")
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n-1:
            possible_actions.append("down")

        #choose a random value from the list of possible actions and map it using the action dictionary
        action = actions[possible_actions[r(0,len(possible_actions) - 1)]]

    else:
        #q-Table
        #returns minimum value of the actions in the current state
        m = np.min(Q[current_state])

        #if the action is possible append the Q-value of the table to the possible actions array.
        # If not append a value indicating a high penalty because it is impossible
        if current_pos[0] != 0:
            possible_actions.append(Q[current_state,0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n-1:
            possible_actions.append(Q[current_state,1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0:
            possible_actions.append(Q[current_state,2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n-1:
            possible_actions.append(Q[current_state,3])
        else:
            possible_actions.append(m - 100)
        # action = np.argmax(possible_actions)

        #sets action to the index of the action the yields the most reward. Chooses a random one if they are all the max-
        #especially at the start of the program where everyting is zero
        action = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)])
        return action

def episode():
    global current_pos,epsilon
    #setting current state to a 49 version of position
    current_state = states[(current_pos[0],current_pos[1])]

    #choosing next action
    action = select_action(current_state)

    #moves player current postion based on next action
    if action == 0:
        current_pos[0] -= 1
    elif action == 1:
        current_pos[0] += 1
    elif action == 2:
        current_pos[1] -= 1
    elif action == 3:
        current_pos[1] += 1

    #finds new state after move mapping
    new_state = states[(current_pos[0],current_pos[1])]

    #updates Q-table using Bellman equation if the new_state isn't a death state or win state
    if new_state not in terminals:

        #current reward plus the max of next rewards in the table. Doesn't literally search recursively, but ultamitely works that way as the table fills
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] + gamma*(np.max(Q[new_state])) - Q[current_state,action])
    else:
        #Only needs to calculate the current reward since future rewards are impossible
        Q[current_state,action] += alpha*(reward[current_pos[0],current_pos[1]] - Q[current_state,action])
        current_pos = [0,0]
        #everytime it dies/wins it raises the chance that the agent will use the Q-table
        epsilon -= 1e-3

#runs forever
run = True

#previous test to run episodes
for i in range(000):
    episode()

#actual Loop
current_pos = [0,0]
while run:
    #sleep(0.3)

    #drawing board layout
    screen.fill(background)
    layout()

    #draw circle
    pygame.draw.circle(screen,(25,129,230),(current_pos[1]*100 + 50,current_pos[0]*100 + 50),30,0)

    #running the pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    #update display
    pygame.display.flip()

    #runs a new episode
    episode()

pygame.quit()
print(epsilon)
# f = open("Q.txt","w")
# f.write(pickle.dumps(Q))
# f.close()