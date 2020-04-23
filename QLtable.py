import numpy as np
import pandas as pd

class QLearningtable(object):
    def __init__(self,actions, lr=0.01, gamma=0.9, eps = 0.9):
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.qtable = pd.DataFrame( columns= actions, dtype=float)
        self.actions = actions
        self.buyrecord= []

    def act(self, state):
        self.check_state(state)
        if np.random.uniform() < self.eps:
            action = self.qtable.loc[[state],:].idxmax(axis=1)[0]
            #action_space = self.qtable.loc[[state],:].max(axis=1)
            #action = np.random.choice(action_space.idxmax(axis=1))
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, sp, a, r):
        self.check_state(sp)
        q_predict = self.qtable.loc[[sp],a][0]
        done = sp[-1]
        if done == 'True':
            q_target = r
        else:
            q_target = r + self.gamma*self.qtable.loc[[sp],:].max(axis=1)[0]
            #print('qtable updated by ',self.lr*(q_target-q_predict))
        self.qtable.loc[[s],a] += self.lr*(q_target-q_predict)


    def check_state(self,state):
        if state not in self.qtable.index:
            temp = pd.DataFrame([[0]*len(self.actions)], index = [state], columns = self.actions)
            self.qtable = pd.concat([self.qtable, temp])
        #print(self.qtable)

