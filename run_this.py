from QLtable import QLearningtable
import numpy as np
import pandas as pd
from functions import *

#l= 200
#data = np.random.rand(1,l)*100
#d_min, d_max = np.min(data), np.max(data)

data = getStockDataVec('^GSPC')
#print('this is length:', len(data))
d_min, d_max = np.min(data), np.max(data)
n_level = 4
data_lvl = np.around([(i-d_min)/((d_max-d_min)/n_level) for i in data])
data_lvl = data_lvl
#data_lvl = np.append(data_lvl,'True')
#print('this is data1 ', data[1], 'this is lvl', data_lvl[2], d_min, d_max)
actions = [1,2,3]
RL = QLearningtable(actions)
e_num = 100

windows_size =5
start_time = 5

for i in range(e_num):
    state = get_state(data_lvl, windows_size, start_time)
    total_profit = 0
    RL.buyrecord = []

    for t in range(start_time,len(data_lvl)):
        action = RL.act(state)
        sp = get_state(data_lvl, windows_size, t)
        #print('action is ', action)
        reward = 0
        if action == 1: #buy
            #print(t, data)
            RL.buyrecord.append(data[t])
        elif action == 2 and len(RL.buyrecord)>0 : #sell
            bought_price = RL.buyrecord.pop(0)
            #reward = max(data[t]-bought_price,0)
            reward = data[t] - bought_price
            RL.learn(state, sp, action, reward)
            total_profit += data[t]-bought_price
        state = sp
        if sp[-1] == 'True':
            print('End time and totoal profit is ', total_profit)
#print('Final table max is ', RL.qtable.max())
#print('Totoal profit is ', total_profit)


