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
import numpy as np

'''
True 140
True 22
True 249
True 39
True 99
True 138
True 89
True 118
True 58
True 63
'''

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10
TRAINING = False
NUM_EPISODES_TRAINING = 50

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        # REINFORCE
        self.n_features = 3 #feature 0 is d/500, feature 1 is v/10 (Scaling the features), feature 2 is 1(bias)
        self.n_actions = 5

        # self.weights = np.zeros((self.n_actions, self.n_features))
        if TRAINING:
            self.weights = np.array([[-4,2,1], [-2,1,0.5], [0,0,0], [2,-1,-0.5], [4,-2,1]])
        else:
            self.weights = np.array([[-3.9256591,   2.04833057, -0.24133139],[-1.915077  ,  1.00875172, -1.05917331],[-0.60547253,  0.03686779, -1.66538722],[ 0.09189934, -1.22377253, -2.2845131 ],[ 0.5837413 , -2.82718085, -2.52153413]])
        self.history = []
        self.rewards = []
        
        super().__init__()

    def getFeatures(self, state):
        x = state[0]
        y = state[1]
        v = state[2]
        theta = state[3]
        if theta > 180:
            theta = theta-360
        desired_theta = 180*math.atan2(0-y,350-x)/math.pi
        up_theta = 180*math.atan2(50-y,350-x)/math.pi
        down_theta = 180*math.atan2(-50-y,350-x)/math.pi
        if theta - desired_theta > 0:
            if theta - desired_theta > 180:
                turn = "left"
            else:
                turn = "right"
        else:
            if theta - desired_theta < -180:
                turn = "right"
            else:
                turn = "left"
        if theta >= 0:
            if theta >= 90:
                d = min(350-y,350+x)
            else:
                d = min(350-y,350-x)
        else:
            if theta < -90:
                d = min(350+y,350+x)
            else:
                d = min(350+y,350-x)
        d = d/500
        v = v/10
        if theta < up_theta and theta > down_theta:
            slow_down = False
        else:
            slow_down = True
        return [turn, slow_down, d, v]

    def alpha(self):
        return 1e-3

    def updateWeight(self):
        G = np.sum(self.rewards)
        # print(G)
        for t in range(len(self.history)):
            if self.history[t][2]: # if slow down
                state = self.history[t][0]
                action = self.history[t][1]
                features = self.getFeatures(state)
                features = np.array([features[2], features[3], 1])
                sa = self.weights@features
                exp = np.exp(sa-np.max(sa))
                exp = exp/np.sum(exp)
                mask = np.zeros(self.n_actions)
                mask[action] = 1
                J = -exp*exp + exp*mask
                if exp[action] != 0:
                    self.weights += self.alpha()*G*(J[:,None]@features[None,:])/exp[action]
            G -= self.rewards[t]

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """
        features = self.getFeatures(state)
        turn = features[0]
        slow_down = features[1]
        features = np.array([features[2], features[3], 1])
        if turn == 'left':
            action_steer = 2
        else:
            action_steer = 0
        if slow_down:
            sa = self.weights@features
            exp = np.exp(sa-np.max(sa))
            exp = exp/np.sum(exp)
            action_acc = np.random.choice(self.n_actions, 1, p=exp)[0]
        else:
            action_acc = 4
        action = np.array([action_steer,action_acc])
        self.history.append((state, action_acc, slow_down))
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
        for e in range(NUM_EPISODES if not TRAINING else NUM_EPISODES_TRAINING):
        
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
                self.rewards.append(reward)
                fpsClock.tick(FPS)

                cur_time += 1
                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))
            if TRAINING:
                self.updateWeight()
                print(self.weights)
            self.history = []
            self.rewards = []

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.mu = 40
        self.sigma = 1
        self.thresh = 50
        # self.lam = 0.1
        self.inertia_count = 0
        self.max_inertia = 10
        self.prev_turn = "left"
        self.margin = 10
        super().__init__()

    def registerObstacles(self, cen_list):
        self.left_surface_list = []
        self.right_surface_list = []
        self.bottom_surface_list = []
        self.top_surface_list = []
        for cen in cen_list: #bottom left corner first
            self.left_surface_list.append((cen[0]-50,cen[1]-60,cen[0]-50,cen[1]+60,False))
            self.right_surface_list.append((cen[0]+50,cen[1]-60,cen[0]+50,cen[1]+60,False))
            self.bottom_surface_list.append((cen[0]-60,cen[1]-50,cen[0]+60,cen[1]-50,False))
            self.top_surface_list.append((cen[0]-60,cen[1]+50,cen[0]+60,cen[1]+50,False))
        # self.left_surface_list.append((350,-350,350,350))
        self.right_surface_list.append((-350,-350,-350,350,True))
        self.bottom_surface_list.append((-350,350,350,350,True))
        self.top_surface_list.append((-350,-350,350,-350,True))
        self.left_surface_list.append((350,100,350,350,True))
        self.left_surface_list.append((350,-350,350,-100,True))
        # self.surface_list = []
        # for cen in cen_list: #bottom left corner first
        #     self.surface_list.append((cen[0]-50,cen[1]-50,cen[0]-50,cen[1]+50))
        #     self.surface_list.append((cen[0]+50,cen[1]-50,cen[0]+50,cen[1]+50))
        #     self.surface_list.append((cen[0]-50,cen[1]-50,cen[0]+50,cen[1]-50))
        #     self.surface_list.append((cen[0]-50,cen[1]+50,cen[0]+50,cen[1]+50))
        # # self.surface_list.append((350,-350,350,350)) skipped
        # self.surface_list.append((-350,-350,-350,350))
        # self.surface_list.append((-350,350,350,350))
        # self.surface_list.append((-350,-350,350,-350)) 
        # print(self.left_surface_list)
        # print(self.right_surface_list)
        # print(self.bottom_surface_list)
        # print(self.top_surface_list)


    def getFeatures(self, state):
        x = state[0]
        y = state[1]
        v = state[2]
        theta = state[3]
        if theta > 180:
            theta = theta-360
        desired_theta = 180*math.atan2(0-y,350-x)/math.pi
        slowDown = False
        turn = None
        
        for sur in self.left_surface_list:
            if sur[0] >= x and sur[1] <= y and y <= sur[3]:
                if theta <= 90 + self.margin and theta >= -90 - self.margin:
                    dist = abs(sur[0]-x)
                    if dist > self.thresh:
                        continue
                    # print("LEFT-----------------")
                    theta1 = 180*math.atan2(sur[1]-y,sur[0]-x)/math.pi
                    theta2 = 180*math.atan2(sur[3]-y,sur[2]-x)/math.pi
                    theta1, theta2 = max(theta1, theta2), min(theta1, theta2)
                    if theta <= theta1 and theta >= theta2:
                        # print("SLOWDOWN--------------")
                        slowDown = True
                    if not sur[4]:
                        if theta > 0:
                            # turn = "left"
                            desired_theta = 90
                        else:
                            # turn = "right"
                            desired_theta = -90
        for sur in self.right_surface_list:
            if sur[0] <= x and sur[1] <= y and y <= sur[3]:
                if theta >= 90 - self.margin or theta <= -90 + self.margin :
                    dist = abs(sur[0]-x)
                    if dist > self.thresh:
                        continue
                    # print("RIGHT-----------------")
                    theta1 = 180*math.atan2(sur[1]-y,sur[0]-x)/math.pi
                    theta2 = 180*math.atan2(sur[3]-y,sur[2]-x)/math.pi
                    theta1, theta2 = max(theta1, theta2), min(theta1, theta2)
                    if theta >= theta1 and theta <= theta2:
                        # print("SLOWDOWN--------------")
                        slowDown = True
                    if not sur[4]:
                        if theta > 0:
                            # turn = "right"
                            desired_theta = 90
                        else:
                            # turn = "left"
                            desired_theta = -90
        for sur in self.bottom_surface_list:
            if sur[1] >= y and x <= sur[2] and sur[0] <= x:
                if theta <= -180 + self.margin or theta >= 0 - self.margin:
                    dist = abs(sur[1]-y)
                    if dist > self.thresh:
                        continue
                    # print("BOTTOM-----------------")
                    theta1 = 180*math.atan2(sur[1]-y,sur[0]-x)/math.pi
                    theta2 = 180*math.atan2(sur[3]-y,sur[2]-x)/math.pi
                    theta1, theta2 = max(theta1, theta2), min(theta1, theta2)
                    if theta <= theta1 and theta >= theta2:
                        # print("SLOWDOWN--------------")
                        slowDown = True
                    if not sur[4]:
                        if theta > 90:
                            # turn = "left"
                            desired_theta = 180
                        else:
                            # turn = "right"
                            desired_theta = 0
        for sur in self.top_surface_list:
            if sur[1] <= y and x <= sur[2] and sur[0] <= x:
                if theta <= 0 + self.margin or theta >= 180 - self.margin:
                    dist = abs(sur[1]-y)
                    if dist > self.thresh:
                        continue
                    # print("TOP-----------------")
                    theta1 = 180*math.atan2(sur[1]-y,sur[0]-x)/math.pi
                    theta2 = 180*math.atan2(sur[3]-y,sur[2]-x)/math.pi
                    theta1, theta2 = max(theta1, theta2), min(theta1, theta2)
                    if theta <= theta1 and theta >= theta2:
                        # print("SLOWDOWN--------------")
                        slowDown = True
                    if not sur[4]:
                        if theta < -90:
                            # turn = "right"
                            desired_theta = 180
                        else:
                            # turn = "left"
                            desired_theta = 0
        
        if theta - desired_theta > 0:
            if theta - desired_theta > 180:
                turn = [1,0,-1]
            else:
                turn = [-1,0,1]
        else:
            if theta - desired_theta < -180:
                turn = [-1,0,1]
            else:
                turn = [1,0,-1]
        # acc = np.array(acc)
        if slowDown:
            acc = 1
        else:
            acc = 3
        return np.argmax(turn), acc

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        
        heading, acc = self.getFeatures(state)
        if heading == 0:
            action_steer = 2
        elif heading == 2:
            action_steer = 0
        else:
            action_steer = 1
        action = np.array([action_steer,acc])

        return action

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
            self.registerObstacles(ran_cen_list)
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
