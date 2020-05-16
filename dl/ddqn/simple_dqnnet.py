from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from dl.ddqn.loss_function import huberloss


class SimpleNNet:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.input_size = state_size
        self.output_size = action_size

        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=self.input_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(self.output_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

