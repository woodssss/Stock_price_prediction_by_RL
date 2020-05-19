from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

class AC(object):
    def __init__(self, lr, gamma, state_size, action_size=3):
        self.gamma = gamma
        self.lr = lr
        self.action_size = action_size
        self.state_size = state_size
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.actor_train, self.actor_predict = self.build_actor()
        self.critic = self.build_critic()
        self.action_space = [i for i in range(self.action_size)]
        self.mdl_name = 'my_temp_mdl.h5'
        self.buyrecord=[]
        self.eps=0.99
        self.G = 0


    def build_actor(self):
        # The actor is almost same as PG, the only difference is the weight of gradient G
        real_input = Input(shape=(self.state_size,))
        G = Input(shape=[1])

        l1 = Dense(16, kernel_initializer='random_uniform', activation='relu')(
            real_input)  # it always takes real_input, G can be viewed as parameter
        l2 = Dense(32, kernel_initializer='random_uniform', activation='relu')(l1)
        actions = Dense(self.action_size, kernel_initializer='random_uniform', activation='softmax')(l2)

        def my_loss(y_true, y_pred):
            log_like = y_true * K.log(K.clip(y_pred, 1e-8, 1 - 1e-8))
            return K.sum(-log_like * G)

        actor_train = Model(input=[real_input, G], output=[actions])

        actor_train.compile(loss=my_loss, optimizer=Adam(lr=self.lr))

        actor_predict = Model(input=[real_input], output=[actions])

        return actor_train, actor_predict

    def build_critic(self):
        real_input = Input(shape=(self.state_size,))
        l1 = Dense(16, kernel_initializer='random_uniform', activation='relu')(
            real_input)  # it always takes real_input, G can be viewed as parameter
        l2 = Dense(32, kernel_initializer='random_uniform', activation='relu')(l1)
        V_val = Dense(1,activation= 'linear')(l2)
        critic = Model(input=[real_input], output=[V_val])
        critic.compile(loss='mean_squared_error', optimizer=Adam(lr=self.lr))
        return critic

    def act(self, state):
        state = state[np.newaxis, :]
        action_prob = self.actor_predict.predict(state)[0]
        # if np.random.uniform()<self.eps:
        #     action = np.random.choice(self.action_space, p = action_prob)
        # else:
        #     action = np.random.choice(self.action_space)
        return np.random.choice(self.action_space, p=action_prob)

    def learn(self, s, a, r, sp):
        # It's different from PG, In PG, need to run through entire episode, here we can update two nn
        # every step
        s = s[np.newaxis, :]
        sp = sp[np.newaxis, :]
        V_s = self.critic.predict(s)
        V_sp = self.critic.predict(sp)

        target = r + self.gamma * V_sp
        G = target - V_s

        one_hot_actions = tf.one_hot(a, self.action_size)
        actions = np.zeros([1, self.action_size])
        actions[np.arange(1), a] = 1
        self.critic.train_on_batch([s],target)
        self.actor_train.train_on_batch([s,G], actions)










