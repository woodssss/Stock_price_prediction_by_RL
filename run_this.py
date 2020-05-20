from QLtable import QLearningtable
import numpy as np
import pandas as pd
from functions import *
import sys
import matplotlib.pyplot as plt

stock_key, windows_size, n_level, e_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
data = getStockDataVec(stock_key)
d_min, d_max = np.min(data), np.max(data)
data_lvl = np.around([(i-d_min)/((d_max-d_min)/n_level) for i in data])
actions = [1,2,3]
RL = QLearningtable(actions)
start_time = windows_size

for i in range(e_num):
    state = get_state_tuple(data_lvl, windows_size, start_time)
    total_profit = 0
    RL.buyrecord = []

    for t in range(start_time,len(data_lvl)):
        action = RL.act(state)
        sp = get_state_tuple(data_lvl, windows_size, t)
        reward = 0
        if action == 1: #buy
            RL.buyrecord.append(data[t])
        elif action == 2 and len(RL.buyrecord)>0 : #sell
            bought_price = RL.buyrecord.pop(0)
            reward = data[t] - bought_price
            RL.learn(state, sp, action, reward)
            total_profit += data[t]-bought_price
        state = sp
        if sp[-1] == 'True':
            print('End time and totoal profit is ', total_profit)

time = [i for i in range(len(data_lvl))]
buy_time = [i for i in range(start_time,len(data_lvl))]
sell_time = [i for i in range(start_time,len(data_lvl))]
agent_action = []
for t in range(start_time,len(data_lvl)):
    s = get_state_tuple(data_lvl, windows_size, t)
    temp_action = RL.act(s)
    agent_action.append(temp_action)
buy_action = data[start_time:].copy()
sell_action = data[start_time:].copy()
for i in range(len(buy_action)):
    if agent_action[i] != 1:
        buy_action[i] = 0
    if agent_action[i] !=2:
        sell_action[i] = 0



plt.plot(time, data, 'b-', label = 'price')
plt.plot(buy_time,buy_action,'ro',label = 'buy')
plt.plot(sell_time,sell_action,'k^',label = 'sell')
plt.legend()
plt.show()
