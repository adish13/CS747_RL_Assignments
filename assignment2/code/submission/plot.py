import numpy as np
import matplotlib.pyplot as plt

class gen_cricket_mdp():
    def __init__(self,statefile_path,parameters,q,write_to):
        self.statefile = statefile_path
        self.paramsfile = parameters
        self.q = q 
        self.write_to = write_to
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
                            # strike = (self.runs[l]%2)^(j%6 == 1)
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
        with open(self.write_to,"w") as f:

            f.write("numStates " + str(self.num_states) + "\n")
            f.write("numActions " + str(self.num_actions) + "\n")
            f.write("end ")
            for i in range(len(self.end_states)-1):
                f.write(str(self.end_states[i]) + ' ')
            f.write(str(self.end_states[-1])+"\n")
            for i in range(self.num_states):
                for j in range(self.num_actions):
                    for k in range(self.num_states):
                        if(self.transit[i,j,k] != 0):
                            f.write("transition "+str(i) + " " +str(j) + " " + str(k) + " " +str(self.reward[i,j,k]) + " " + str(self.transit[i,j,k]) + "\n")

            f.write("mdptype " + str(self.mdptype) + "\n")
            f.write("discount "+ str(self.discount)+"\n")


class MDP_solver():
    def __init__(self,mdp_path,policyfile='-1'):
        self.path = mdp_path
        self.mdp = dict()
        self.read_mdp()
        self.v_star = np.zeros(self.mdp['numStates'])
        self.pi_star = np.zeros(self.mdp['numStates'],dtype=int)
        self.policyfile = policyfile
        self.win_prob = 0
        if(self.policyfile == '-1'):
            self.alg_vi()

        else:
            self.eval_policy()

    def eval_policy(self):
        count = 0
        with open(self.policyfile) as file:
            while(line := file.readline().rstrip()):
                action = int(line)
                if(action == 6):
                    action = 4
                elif(action == 4):
                    action = 3
                self.pi_star[count] = action
                count += 2
        transit_pi = self.mdp['transition'][np.arange(self.mdp["numStates"]), self.pi_star]
        reward_pi = self.mdp["reward"][np.arange(self.mdp["numStates"]), self.pi_star]
        
        # V = (I - gamma T)^-1 @ (expected_rpi)
        expected_rpi = np.sum(transit_pi*reward_pi,axis=1,keepdims=True)
        i_minus_gT= np.eye(self.mdp["numStates"]) - self.mdp["gamma"] * transit_pi
        self.v_star = np.linalg.inv(i_minus_gT)@expected_rpi

        # converting from shape (n,1) to shape (n,) 
        self.v_star = np.squeeze(self.v_star)
        # print(self.v_star)

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
    # def alg_hpi(self):
    #     pi = np.random.randint(low=0, high=self.mdp["numActions"], size=self.mdp["numStates"])
    #     temp_pi = pi
    #     while (True):

    #         transit_pi = self.mdp['transition'][np.arange(self.mdp["numStates"]), temp_pi]
    #         reward_pi = self.mdp["reward"][np.arange(self.mdp["numStates"]), temp_pi]
            
    #         # V = (I - gamma T)^-1 @ (expected_rpi)
    #         expected_rpi = np.sum(transit_pi*reward_pi,axis=1,keepdims=True)
    #         i_minus_gT= np.eye(self.mdp["numStates"]) - self.mdp["gamma"] * transit_pi
    #         self.v_star = np.linalg.inv(i_minus_gT)@expected_rpi

    #         # converting from shape (n,1) to shape (n,) 
    #         self.v_star = np.squeeze(self.v_star)

    #         self.pi_star = np.argmax(np.sum(self.mdp["transition"] * (self.mdp["reward"] + self.mdp["gamma"] * self.v_star), axis=-1), axis=-1)
    #         if np.array_equal(self.pi_star, temp_pi):
    #             break
    #         temp_pi = self.pi_star



    def print_output(self):
        return self.v_star[0]



def gen_statefile(balls,runs):
    with open("plots/statefile_" + str(balls)+str(runs).zfill(2),"w") as f:
        for b in range(balls):
            for r in range(runs):
                for player in range(1):
                    f.write(str(balls-b).zfill(2)+  str(runs-r).zfill(2))
                    f.write("\n")

# Analysis 1
optimum_win_prob = []
pol_win_prob = []

q = 0
gen_statefile(15,30)
statefile_name = "plots/statefile_1530"
for i in range(11):
    mdp_name = "plots/cmdp_1530_" + str(q)
    gen_cricket_mdp(statefile_name,"data/cricket/sample-p1.txt",q,mdp_name)
    optimum_win_prob.append(MDP_solver(mdp_name).print_output())
    pol_win_prob.append(MDP_solver(mdp_name,"data/cricket/rand_pol.txt").print_output())    
    q += 0.1

x = np.arange(0,1.1,0.1)
plt.plot(x,optimum_win_prob,label = "Optimum policy")
plt.plot(x,pol_win_prob,label='Random policy')
plt.xlabel("q value")
plt.ylabel("win probability")
plt.title("Analysis Part 1")
plt.legend()
plt.savefig("varying_q.png")
plt.close()

# Analysis 2
optimum_win_prob2 = []
pol_win_prob2 = []

def gen_random_policy(max_balls,max_runs,):
    
    with open("rand_pol.txt") as file:
        f = open("plots/rand_pol_" + str(max_balls)+str(max_runs).zfill(2),"w")
        while (line := file.readline().rstrip()):
            words = line.split()
            state = words[0]
            balls = int(state[:2])
            runs = int(state[2:])
            if(balls <= max_balls and runs <= max_runs):
                f.write(words[1])
                f.write("\n")

for i in range(1,21):
    gen_statefile(10,i)
    statefile_name = "plots/statefile_" + str(10)+str(i).zfill(2)
    mdp_name = "plots/cmdp_"+str(10)+str(i).zfill(2)
    gen_cricket_mdp(statefile_name,"data/cricket/sample-p1.txt",0.25,mdp_name)
    optimum_win_prob2.append(MDP_solver(mdp_name).print_output())
    gen_random_policy(10,i)
    random_pol_file = "plots/rand_pol_" + str(10)+str(i).zfill(2)
    pol_win_prob2.append(MDP_solver(mdp_name,random_pol_file).print_output())

x = np.arange(1,21,1)
plt.plot(x,optimum_win_prob2,label = "Optimum policy")
plt.plot(x,pol_win_prob2,label='Random policy')
plt.xlabel("Number of runs")
plt.ylabel("win probability")
plt.title("Analysis Part 2")
plt.legend()
plt.savefig("varying_runs.png")
plt.close()

# Analysis 3
optimum_win_prob3 = []
pol_win_prob3 = []

for i in range(1,16):
    gen_statefile(i,10)
    statefile_name = "plots/statefile_" + str(i)+str(10).zfill(2)
    mdp_name = "plots/cmdp_"+str(i)+str(10).zfill(2)
    gen_cricket_mdp(statefile_name,"data/cricket/sample-p1.txt",0.25,mdp_name)
    optimum_win_prob3.append(MDP_solver(mdp_name).print_output())
    gen_random_policy(i,10)
    random_pol_file = "plots/rand_pol_" + str(i)+str(10).zfill(2)
    pol_win_prob3.append(MDP_solver(mdp_name,random_pol_file).print_output())

x = np.arange(1,16,1)
plt.plot(x,optimum_win_prob3,label = "Optimum policy")
plt.plot(x,pol_win_prob3,label='Random policy')
plt.xlabel("Number of balls")
plt.ylabel("Win probability")
plt.title("Analysis Part 3")
plt.legend()
plt.savefig("varying_balls.png")
plt.close()
