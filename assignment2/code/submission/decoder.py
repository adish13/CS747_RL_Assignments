import argparse,sys
parser = argparse.ArgumentParser()

class Decoder():
    def __init__(self,statefile,vpfile):
        self.statefile = statefile
        self.vpfile = vpfile
        self.states = dict()
        self.value_policy = []
        self.actions =[0,1,2,4,6]
        self.line_number = 0

        self.read_statefile()
        self.read_vpfile()
        self.decode()

    def read_statefile(self):
        with open(self.statefile) as file:
            while( line := file.readline().rstrip()):
                self.states[self.line_number] = line
                self.line_number += 2

    def read_vpfile(self):
        with open(self.vpfile) as file:
            while( line := file.readline().rstrip()):
                words = line.split()
                value = float(words[0])
                policy = int(words[1])
                self.value_policy.append((policy,value))

    def decode(self):
        for i in range(len(self.value_policy)):
            if i in self.states:
                print(self.states[i],self.actions[self.value_policy[i][0]], "{:.6f}".format(self.value_policy[i][1]))

parser.add_argument("--value-policy",type=str,dest='value_policy')
parser.add_argument("--states",type=str)
args = parser.parse_args()

Decoder(args.states,args.value_policy)


