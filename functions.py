import numpy as np
import math
import os



def getStockDataVec(key):
    curpath = os.getcwd()
    vec=[]
    lines = open(curpath + '/data/' + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    return vec


def sigmoid(x):
    return 1/(1+math.exp(-x))

def get_state(data, wd, t):
    st = t-wd
    done = 'True' if t==len(data)-1 else 'False'
    block = data[st:t]
    block = np.append(block, done)
    return tuple(block)
