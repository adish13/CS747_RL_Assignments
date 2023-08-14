"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # START EDITING HERE
        # You can add any other variables you need here

        ### Code using random X arms (discarded since bandit instance already random)
        # self.chosen_arms = np.random.permutation(num_arms)[:num_arms//200]
        # self.successes = dict.fromkeys(self.chosen_arms,0)
        # self.failures = dict.fromkeys(self.chosen_arms,0)
        # self.beta_samples = dict.fromkeys(self.chosen_arms,0)

        # Code using the first X arms directly
        self.fraction = int(np.sqrt(num_arms))
        self.chosen_arms = np.arange(self.fraction)
        self.successes = np.zeros(self.fraction)
        self.failures = np.zeros(self.fraction)
        self.beta_samples = np.zeros(self.fraction)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        ### Code using random X arms (discarded since bandit instance already random)
        # for i in range(len(self.chosen_arms)):
        #     self.beta_samples[self.chosen_arms[i]] = np.random.beta(self.successes[self.chosen_arms[i]]+1,self.failures[self.chosen_arms[i]]+1)

        # return max(self.beta_samples, key= lambda x: self.beta_samples[x])

        ### Code using the first X arms directly
        self.beta_samples = [np.random.beta(self.successes[i]+1,self.failures[i]+1) for i in range(self.fraction)]
        return np.argmax(self.beta_samples)
        # END EDITING HERE

    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1
        # END EDITING HERE
