from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

class DQN(object):
    def __init__(self, lr, gamma, state_size, mem_size, action_size=3):
        self.gamma = gamma
        self.lr = lr
        self.action_size = action_size
        self.state_size = state_size
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size, self.state_size))
        self.state_prime_memory = np.zeros((self.mem_size, self.state_size))
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size)
        self.action_space = [i for i in range(self.action_size)]
        self.mdl_name = 'my_temp_mdl.h5'
        self.buyrecord=[]
        self.eps=0.99


        self.dqn_eval = self.build_dqn()

        self.dqn_target = self.build_dqn()

    def build_dqn(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.state_size,)))
        model.add(Dense(32, activation='relu', input_shape=(64,)))
        model.add(Dense(self.action_size))

        model.compile(loss='mse', optimizer=Adam(lr = self.lr))
        return model
    def trasition_store(self,s, a, r, sp, step):
        index = step % self.mem_size if step > self.mem_size else step
        self.state_memory[index-1,:] = s
        self.action_memory[index-1] = a
        self.reward_memory[index-1] = r
        self.state_prime_memory[index-1,:] = sp

    def act(self,state):
        if np.random.random() < self.eps:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([state], copy=False, dtype=np.float32)
            actions = self.dqn_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        # Replace target net by latest eval net
        self.dqn_target.set_weights(self.dqn_eval.get_weights())

        q_eval = self.dqn_eval.predict(self.state_memory)

        q_next = self.dqn_target.predict(self.state_prime_memory)

        q_target = q_eval[:]
        indices = np.arange(self.mem_size)
        q_target[indices, self.action_memory] = self.reward_memory + \
                                    self.gamma * np.max(q_next, axis=1)
        self.dqn_eval.train_on_batch(self.state_memory, q_target)


