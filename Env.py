# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q ]
        self.action_space.append(tuple((0, 0))) # (0, 0) tuple that represents ’no-ride’ action
                    
        self.state_space = [(x,time,day) for x in range(m)for day in range(d) for time in range(t) ] 
        
        self.state_init = (np.random.randint(m),np.random.randint(t),np.random.randint(d))

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros(m+t+d, dtype = int)
        
        for i in enumerate(state):
        print(i)
        if i[0] == 0:
            state_encod[i[1]] = 1
        elif i[0] == 1:
            state_encod[m+i[1]] = 1
        elif i[0] == 2:
            state_encod[m+t+i[1]] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        if action[0] == 0 and action[1] == 0:
            reward = -C
        else:
            time_p_to_q = Time_matrix[action[0],action[1],state[1],state[2]]
            time_x_to_p = Time_matrix[state[0],action[0],state[1],state[2]]
            
            reward = R*time_p_to_q - C*(time_p_to_q + time_x_to_p)
            
        return reward



    def next_state_func(self, state, action, Time_matrix):
    """Takes state and action as input and returns next state"""
    
    new_loc= state[0]
    new_time=state[1]
    new_day=state[2]
    
    if action[0] == 0 and action[1] == 0: # (0, 0) tuple that represents ’no-ride’ action
        new_time += 1
        if (new_time == t ):
            new_time= 0
            new_day += 1
            if (new_day == d):new_day = 0
            
        next_state=(new_loc,new_time,new_day)
        
    else: # When the driver chooses an action (p,q)
        time_x_to_p = Time_matrix[state[0],action[0],state[1],state[2]]
        time_p_to_q = Time_matrix[action[0],action[1],state[1],state[2]]
        total_trans_time = time_x_to_p + time_p_to_q

        if (new_time + total_trans_time > (t-1)): # if total time  > 23 (time index start from 0)
            new_time = ( total_trans_time - ((t-1)-new_time) - 1 )
            new_day += 1
            if (new_day == d):new_day = 0 # reset day
        else:
            new_time = new_time + total_trans_time
        next_state=(action[1],new_time,new_day)

    return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
