from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam

from tengu.drlfx.base_rl.loss_function import huberloss
import numpy as np


class SimpleNNet:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.input_size = state_size
        self.output_size = action_size

        main_input = Input(shape=(self.input_size,), name='main_input')
        model = Dense(hidden_size, activation='relu')(main_input)
        model = Dense(hidden_size, activation='relu')(model)
        model = Dense(self.output_size, activation='linear')(model)

        self._model = Model(inputs=main_input, outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def predict(self, x):
        x = np.reshape(x, [len(x), self.input_size])
        return self._model.predict(x)

    def train_on_batch(self, x, y):
        x = np.reshape(x, [len(x), self.input_size])
        y = np.reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch(x, y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()
