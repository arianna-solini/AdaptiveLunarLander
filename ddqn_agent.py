import numpy as np
import random
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear



class DDQN:

    """ Double Deep Q Learning agent """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = 0.99
        self.batch_size = 128
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.epsilon_decay = 0.996
        self.memory = deque(maxlen=1000000)
        self.target_network = self.build_model()
        self.dqn_network = self.build_model()
        #self.TAU = 0.1  # for soft update


    def build_model(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_space, activation=relu))#150
        model.add(Dense(32, activation=relu))#120
        model.add(Dense(self.action_space, activation=linear))  # output layer = action space
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act_greedy_policy(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.dqn_network.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):

        if len(self.memory) < self.batch_size:
            return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        update_input = np.array([i[0] for i in minibatch])
        action = np.array([i[1] for i in minibatch])
        reward = np.array([i[2] for i in minibatch])
        update_target = np.array([i[3] for i in minibatch])
        done = np.array([i[4] for i in minibatch])

        update_input = np.squeeze(update_input)
        update_target = np.squeeze(update_target)

        target = self.dqn_network.predict(update_input)
        #target_next = self.dqn_network.predict_on_batch(update_target)
        #target_val = self.target_network.predict_on_batch(update_target)
        next_q_values = self.target_network.predict(update_target)[
            range(self.batch_size), np.argmax(self.dqn_network.predict(update_target), axis=1)]
        target[range(self.batch_size), action] = reward + (1 - done) * next_q_values * self.gamma

        self.dqn_network.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    ## Uncomment this to try the soft_update approach ##
    '''def soft_update_weights(self):
        q_network_theta = self.dqn_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta, target_network_theta):
            target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)'''

    def copy_weights(self):
        self.target_network.set_weights(self.dqn_network.get_weights())

