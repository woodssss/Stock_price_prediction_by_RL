from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import TensorBoard
from time import time
import numpy as np
import tensorflow as tf

class PG(object):
    def __init__(self, lr, gamma, state_size, action_size=3):
        self.gamma = gamma
        self.lr = lr
        self.action_size = action_size
        self.state_size = state_size
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.agent_train, self.agent_predict = self.build_model()
        self.action_space = [i for i in range(self.action_size)]
        self.mdl_name = 'my_temp_mdl.h5'
        self.buyrecord=[]
        self.eps=0.99
        self.G = 0

    def build_model(self):
        # for multiple inputs, we can't use Sequential()
        real_input = Input(shape=(self.state_size,))
        G = Input(shape=[1])

        l1 = Dense(16,kernel_initializer='random_uniform', activation='relu')(real_input) # it always takes real_input, G can be viewed as parameter
        l2 = Dense(32,kernel_initializer='random_uniform', activation='relu')(l1)
        actions = Dense(self.action_size,kernel_initializer='random_uniform', activation='softmax')(l2)

        def my_loss(y_true, y_pred):
            log_like = y_true*K.log(K.clip(y_pred, 1e-8, 1-1e-8))
            return K.sum(-log_like*G)


        agent_train = Model(input=[real_input, G], output=[actions])

        agent_train.compile(loss= my_loss, optimizer=Adam(lr=self.lr))

        agent_predict = Model(input=[real_input], output=[actions])

        return agent_train, agent_predict

    def act(self,state):
        state = state[np.newaxis,:]
        action_prob = self.agent_predict.predict(state)[0]
        # if np.random.uniform()<self.eps:
        #     action = np.random.choice(self.action_space, p = action_prob)
        # else:
        #     action = np.random.choice(self.action_space)
        return np.random.choice(self.action_space, p = action_prob)

    def store(self,s,a,r):
        self.state_memory.append(s)
        self.action_memory.append(a)
        self.reward_memory.append(r)

    def learn(self):
        # First, we need to run through the episode and calculate the G value for episode
        state_memory, action_memory, reward_memory = self.state_memory, self.action_memory, self.reward_memory
        #print('this is reward: ', reward_memory)
        G = np.zeros_like(reward_memory)
        for i in range(len(G)):
            G_sum = 0
            discount = 1
            for j in range(i,len(G)):
                G_sum+=reward_memory[j]*discount
                discount*=self.gamma
            G[i] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        self.G = (G-mean)/std
        #print('this is G value: ', self.G)

        one_hot_actions = tf.one_hot(self.action_memory, self.action_size)
        #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        self.agent_train.train_on_batch([state_memory, self.G], one_hot_actions)

        self.state_memory, self.action_memory, self.reward_memory = [], [], []

    def save_mdl(self):
        self.agent_train.save(self.mdl_name)

    def load_mdl(self):
        self.agent_train = load_model(self.mdl_name)








