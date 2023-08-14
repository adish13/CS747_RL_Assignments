#! /usr/bin/python
import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np
import pulp

class MDP_solver():
    def __init__(self,mdp_path,alg,policyfile):
        self.alg = alg
        self.path = mdp_path
        self.mdp = dict()
        self.read_mdp()
        self.v_star = np.zeros(self.mdp['numStates'])
        self.pi_star = np.zeros(self.mdp['numStates'],dtype=int)
        self.policyfile = policyfile
        if(self.policyfile == '-1'):

            if(self.alg == 'vi'):
                self.alg_vi()
            elif(self.alg == 'hpi'):
                self.alg_hpi()
            elif(self.alg == 'lp'):
                self.alg_lp()

            self.print_output()

        else:
            self.eval_policy()
            self.print_output()

    def eval_policy(self):
        count = 0
        with open(self.policyfile) as file:
            while(line := file.readline().rstrip()):
                self.pi_star[count] = int(line)
                count += 1
        
        transit_pi = self.mdp['transition'][np.arange(self.mdp["numStates"]), self.pi_star]
        reward_pi = self.mdp["reward"][np.arange(self.mdp["numStates"]), self.pi_star]
        
        # V = (I - gamma T)^-1 @ (expected_rpi)
        expected_rpi = np.sum(transit_pi*reward_pi,axis=1,keepdims=True)
        i_minus_gT= np.eye(self.mdp["numStates"]) - self.mdp["gamma"] * transit_pi
        self.v_star = np.linalg.inv(i_minus_gT)@expected_rpi

        # converting from shape (n,1) to shape (n,) 
        self.v_star = np.squeeze(self.v_star)

    def read_mdp(self):
        with open(self.path) as file:
            while (line := file.readline().rstrip()):
                words = line.split()
                if(words[0] == 'numStates'):
                    self.mdp['numStates'] = int(words[-1])

                elif(words[0] == 'numActions'):
                    self.mdp['numActions'] = int(words[-1])

                    # Initialise transition and reward arrays
                    num_states = self.mdp['numStates']
                    num_actions = self.mdp['numActions']
                    self.mdp['transition'] = np.zeros((num_states,num_actions,num_states))
                    self.mdp['reward'] = np.zeros((num_states,num_actions,num_states))

                elif(words[0] == 'end'):
                    l = words[1:]
                    self.mdp['end'] = list(map(int,l))

                elif(words[0] == 'transition'):
                    l = words[1:]
                    l = list(map(eval,l))
                    s,a,s_,r,p = l
                    self.mdp['transition'][s,a,s_] = p
                    self.mdp['reward'][s,a,s_] = r 

                elif(words[0] == 'mdptype'):
                    self.mdp['mdptype'] = words[-1]

                elif(words[0] == 'discount'):
                    self.mdp['gamma'] = eval(words[-1])


    def value_iteration(self,V,policy):
        reward_plus_tempv = (self.mdp['reward'] + self.mdp['gamma']*V)
        x = self.mdp['transition']*reward_plus_tempv
        sum_x_along_s_ = np.sum(x,axis = -1)
        if(policy == True):
            return np.argmax(sum_x_along_s_,axis = -1)
        else:
            return np.max(sum_x_along_s_,axis = -1)

    def alg_vi(self):

        temp_v = np.random.randn(self.mdp['numStates'])
        while(True):
            self.v_star = self.value_iteration(temp_v,False)
            if(np.allclose(self.v_star,temp_v,atol= 1e-10,rtol= 0)):
                break
            temp_v = self.v_star

        self.pi_star = self.value_iteration(self.v_star,True)

    def alg_hpi(self):
        pi = np.random.randint(low=0, high=self.mdp["numActions"], size=self.mdp["numStates"])
        temp_pi = pi
        while (True):

            transit_pi = self.mdp['transition'][np.arange(self.mdp["numStates"]), temp_pi]
            reward_pi = self.mdp["reward"][np.arange(self.mdp["numStates"]), temp_pi]
            
            # V = (I - gamma T)^-1 @ (expected_rpi)
            expected_rpi = np.sum(transit_pi*reward_pi,axis=1,keepdims=True)
            i_minus_gT= np.eye(self.mdp["numStates"]) - self.mdp["gamma"] * transit_pi
            self.v_star = np.linalg.inv(i_minus_gT)@expected_rpi

            # converting from shape (n,1) to shape (n,) 
            self.v_star = np.squeeze(self.v_star)

            self.pi_star = np.argmax(np.sum(self.mdp["transition"] * (self.mdp["reward"] + self.mdp["gamma"] * self.v_star), axis=-1), axis=-1)
            if np.array_equal(self.pi_star, temp_pi):
                break
            temp_pi = self.pi_star


    def alg_lp(self):
        v = pulp.LpVariable.dicts("s", (range(self.mdp['numStates']))) 
        prob = pulp.LpProblem("lp_mdp", pulp.LpMinimize) 

        # objective function
        prob += sum ([v[i] for i in range (self.mdp['numStates'])])

        #  constaints
        for i in range (self.mdp['numStates']):
            for a in range(self.mdp['numActions']):
                prob += v[i] >= self.mdp['gamma'] * sum(self.mdp['transition'][i, a, j]* v[j] for j in range(self.mdp['numStates'])) + sum(self.mdp['transition'][i, a, j]*self.mdp['reward'][i, a, j] for j in range(self.mdp['numStates']))
        prob.solve(pulp.apis.PULP_CBC_CMD(msg=0))

        for i in range (self.mdp['numStates']):
            self.v_star[i] = v[i].varValue
        self.pi_star = self.value_iteration(self.v_star,True)

    def print_output(self):
        for i in range(len(self.v_star)):
            print("{:.6f}".format(self.v_star[i]) + ' ' + str(self.pi_star[i]))

# path of input mdp file
parser.add_argument("--mdp",type=str,help='mdp file argument missing. Use --mdp to add it')

# algorithm is either vi, hpi or lp
parser.add_argument("--algorithm",type=str,default='hpi')

# #optional, followed by a policy file
parser.add_argument("--policy",type=str,default='-1')

args = parser.parse_args()
   
MDP_solver(args.mdp,args.algorithm,args.policy)

