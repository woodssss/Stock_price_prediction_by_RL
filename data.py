import numpy as np
import pandas as pd


l= 20

data = np.random.rand(1,l)*100

d_min, d_max = np.min(data), np.max(data)

n_level = 10

data_lvl = np.around([i%((d_max-d_min)/n_level) for i in data])
print(data_lvl)

actions = [1,2,3]

q_table = pd.DataFrame( columns= actions, dtype=float)

#print(q_table.head())

def get_state(data, wd_size, t):
    return tuple(data[0][t:t+wd_size])
s1 = get_state(data_lvl, 3, 3)
s2 = get_state(data_lvl, 3, 5)
print('this is s1 and s2: ', s1, s2)
#print('this is s11: ', (data_lvl[0][3:6]))

def check_state(s,table):
    if s not in table.index:
        temp = pd.DataFrame([[0]*len(actions)], index=[s], columns=table.columns,dtype=float)
        table = pd.concat([table, temp])
    return table

q_table = check_state(s1,q_table)
q_table = check_state(s2,q_table)
print(q_table)