from PG_brain import PG
import numpy as np
import pandas as pd
from functions import *
import sys
import matplotlib.pyplot as plt

stock_key, windows_size, e_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
data = getStockDataVec(stock_key)
RL = PG(lr=0.00001, gamma=0.97, state_size=windows_size)
start_time = windows_size

for i in range(e_num):
    total_profit = 0
    # for each episode, we need to run the entire episode and record everything
    for t in range(start_time,len(data)):
        state = get_state(data, windows_size, t)
        action = RL.act(state)
        reward = 0
        if action == 1:  # buy
            RL.buyrecord.append(data[t])
            #reward = -data[t]
        elif action == 2 and len(RL.buyrecord) > 0:  # sell
            bought_price = RL.buyrecord.pop(0)
            reward = data[t] - bought_price
            total_profit += data[t] - bought_price
        RL.store(state,action,reward)
        if t==len(data)-1:
            print('End time and totoal profit is ', total_profit)

    # after running the episode, we can start training mdl
    RL.learn()


time = [i for i in range(len(data))]
buy_time = [i for i in range(start_time,len(data))]
sell_time = [i for i in range(start_time,len(data))]
agent_action = []
for t in range(start_time,len(data)):
    s = get_state(data, windows_size, t)
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