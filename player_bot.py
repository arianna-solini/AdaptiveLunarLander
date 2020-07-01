from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear

class PLAYER:

    """ A trained agent that plays the game """

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_space, activation=relu))
        model.add(Dense(32, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        return model

