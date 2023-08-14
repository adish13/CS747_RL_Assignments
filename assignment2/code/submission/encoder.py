import argparse
parser = argparse.ArgumentParser()
import numpy as np

class gen_cricket_mdp():
    def __init__(self,statefile_path,parameters,q):
        self.statefile = statefile_path
        self.paramsfile = parameters
        self.q = q 
        self.num_states = 0
        self.num_actions = 5
        self.end_states = []
        self.states = dict()
        self.max_balls = -1
        self.max_runs = -1

        self.read_statefile()

        self.probs = np.zeros((self.num_actions,7))
        self.non_strike_probs = [q,(1-q)/2,(1-q)/2]
        self.read_params()

        self.runs = [0,1,2,3,4,6]

        self.transit = np.zeros((self.num_states,self.num_actions,self.num_states))
        self.reward = np.zeros((self.num_states,self.num_actions,self.num_states))

        self.init_transit()
        self.mdptype = 'episodic'
        self.discount = 1

        self.print_mdp()

    def read_statefile(self):

        with open(self.statefile) as file:
            while( line := file.readline().rstrip()):
                balls = int(line[:2])
                self.max_balls = max(self.max_balls,balls)
                runs = int(line[2:])
                self.max_runs = max(self.max_runs,runs)
                self.states[(balls,runs,0)] = self.num_states
                self.num_states += 1
                self.states[(balls,runs,1)] = self.num_states
                self.num_states += 1 
        # 2 extra states for losing and wining position, both are end states
        # num_states - 1 is losing and num_states - 2 is winning
        self.num_states += 2
        self.end_states = [self.num_states - 2, self.num_states - 1]
    
    def read_params(self):
        count = 0
        first_line = True
        with open(self.paramsfile) as file:
            while( line := file.readline().rstrip()):
                if(first_line == True):
                    first_line = False
                    continue    
                words = line.split()
                for i in range(7):
                    self.probs[count,i] = float(words[i+1])
                count += 1

    def init_transit(self):
        for i in range(1,self.max_runs+1):
            for j in range(1,self.max_balls+1):
                for k in range(self.num_actions):
                    for l in range(6):
                        # f = final runs
                        f = i - self.runs[l]

                        # remaining runs and balls greater than 0, game still continues
                        if(f > 0 and j > 1):
                            strike = self.runs[l]%2
                            if(j%6==1):
                                if(strike == 1):
                                    strike = 0
                                else:
                                    strike = 1
                                    
                            self.transit[self.states[(j,i,0)],k,self.states[(j-1,f,strike)]] += self.probs[k,l+1]

                        # remaining runs more than 0, but no more ball left
                        # transit to losing state, reward = 0
                        elif(f > 0 and j == 1):
                            self.transit[self.states[(j,i,0)],k,self.num_states - 1] += self.probs[k,l+1]
                            self.reward[self.states[(j,i,0)],k,self.num_states - 1] += 0
                        
                        # remaining runs <= 0
                        # transit to winning state, reward = 1
                        elif(f <= 0):
                            self.transit[self.states[(j,i,0)],k,self.num_states - 2] += self.probs[k,l+1]
                            self.reward[self.states[(j,i,0)],k,self.num_states - 2] = 1
                        


                    for l in range(2):
                        # f = final runs
                        f = i - self.runs[l]

                        # remaining runs and balls greater than 0, game still continues
                        if(f > 0 and j > 1):
                            strike = self.runs[l]%2
                            if(j%6==1):
                                if(strike == 1):
                                    strike = 0
                                else:
                                    strike = 1                            
                            strike = 1 - strike
                            self.transit[self.states[(j,i,1)],k,self.states[(j-1,f,strike)]] += self.non_strike_probs[l+1]

                        # remaining runs more than 0, but no more ball left
                        # transit to losing state, reward = 0
                        elif(f > 0 and j == 1):
                            self.transit[self.states[(j,i,1)],k,self.num_states - 1] += self.non_strike_probs[l+1]
                            self.reward[self.states[(j,i,1)],k,self.num_states - 1] += 0
                        
                        # remaining runs <= 0
                        # transit to winning state, reward = 1
                        elif(f <= 0):
                            self.transit[self.states[(j,i,1)],k,self.num_states - 2] += self.non_strike_probs[l+1]
                            self.reward[self.states[(j,i,1)],k,self.num_states - 2] = 1
                    
                    
                    # transit to losing state if got out
                    self.transit[self.states[(j,i,0)],k,self.num_states - 1] += self.probs[k,0]
                    self.reward[self.states[(j,i,0)],k,self.num_states - 1] += 0
                    self.transit[self.states[(j,i,1)],k,self.num_states - 1] += self.non_strike_probs[0]
                    self.reward[self.states[(j,i,1)],k,self.num_states - 1] += 0
     

    def print_mdp(self):
        print("numStates", self.num_states)
        print("numActions", self.num_actions)
        print("end", end=' ')
        for i in range(len(self.end_states)-1):
            print(str(self.end_states[i]), end = ' ')
        print(self.end_states[-1])
        for i in range(self.num_states):
            for j in range(self.num_actions):
                for k in range(self.num_states):
                    if(self.transit[i,j,k] != 0):
                        print("transition",i,j,k,self.reward[i,j,k],self.transit[i,j,k])

        print("mdptype",self.mdptype)
        print("discount",self.discount)


# path of state file
parser.add_argument("--states",type=str, default= '/host/sem5/cs747/assignment2/code/statefile_1520')
parser.add_argument("--parameters",type=str,default='/host/sem5/cs747/assignment2/code/data/cricket/sample-p1.txt')
parser.add_argument("--q",type=float,default=0.25)

args = parser.parse_args()

gen_cricket_mdp(args.states,args.parameters,args.q)
