from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time 
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

# steerings
anti_clockwise = 0
neutral = 1
clockwise = 2

# acceleration

# -5 units
nfast = 0

# -3.95 units
nslow = 1

# 0 units
zero = 2

# 3.95 units
pslow = 3

# 5 units
pfast = 4


class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()
        self.angle_needed = None
        self.direction = None
        self.set = False

    def set_steer(self,cur_angle):
        if(self.angle_needed - cur_angle < 180):
            self.direction = clockwise
        else:
            self.direction = anti_clockwise
        
    def move_staight(self):
        return neutral,pfast
    
    def turn_without_moving(self):
        return self.direction,zero

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        if(self.set == False):
            x,y,vel,angle = state
            angle%= 360
            self.set = True
            angle_in_radians = np.arctan2([-y],[350-x])
            self.angle_needed = ((180)/np.pi)*angle_in_radians
            self.angle_needed %= 360
            self.set_steer(cur_angle=angle)


        threshold = 2.1
        angle = state[3]%360
        if(abs(angle - self.angle_needed) >= threshold):
            action_steer,action_acc = self.turn_without_moving()
        else:
            action_steer,action_acc = self.move_staight()
    
        action = np.array([action_steer, action_acc])  

        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()
                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1
                if terminate:
                    road_status = reached_road
                    self.set = False 
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))


class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        super().__init__()

        # actions
        self.steer = None
        self.acc = None
        self.action = None
        self.pits_list = None
        self.threshold = 3

    def close_to_vpit(self,state):
        for center in self.pits_list:
            x_cor = state[0]
            y_cor = state[1]
            delta = 70
            if(abs(x_cor - center[0]) <= delta):
                if(y_cor > 0):
                    if (y_cor >= center[1] + 55 and center[1] > 0):
                        return True
                else:
                    if (y_cor <= center[1] - 55 and center[1] < 0):
                        return True
        return False

    def close_to_center(self,state):
        y_cor = state[1]
        if(abs(y_cor) <= 30):
            return True
        else:
            return False

    def drive_vertical(self, state):
        angle = state[3]
        y_cor = state[1]
        if( ( abs(angle - 270) <= self.threshold  and y_cor > 0)  or (abs(angle-90) <= self.threshold and y_cor < 0)):
            self.steer = neutral
            self.acc = pfast
        elif(y_cor > 0):
            self.acc = nfast
            if (angle >= 90 and angle <= 270):
                self.steer = clockwise
            else:
                self.steer  = anti_clockwise
        else:
            self.acc = nfast
            if (angle >= 90 and angle <= 270):
                self.steer = anti_clockwise
            else:
                self.steer  = clockwise

        self.action = np.array([self.steer, self.acc])
        return self.action
    
    def drive_right(self, state):
        angle = state[3]
        if(angle <= 0 + self.threshold or angle >= 360 - self.threshold):
            self.steer = neutral
            self.acc = pfast
        else:
            self.acc = nfast
            if(angle >= 180):
                self.steer = clockwise
            else:
                self.steer = anti_clockwise    

        self.action = np.array([self.steer, self.acc])          
        return self.action

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        # WE are at center just go straight
        state[3] %= 360
        if(self.close_to_center(state)):
            return self.drive_right(state)

        return self.drive_right(state) if(self.close_to_vpit(state)) else self.drive_vertical(state)



    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]      
            self.pits_list = ran_cen_list      
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)