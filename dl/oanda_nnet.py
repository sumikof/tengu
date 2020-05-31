from keras import Model, Input
from keras.layers import Dense, Concatenate, Lambda, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, LSTM
from keras.optimizers import Adam
import keras.backend as k
import numpy as  np
from dl.base_rl.loss_function import huberloss
from dl.test_oanda import StepState, ACTION_SIZE


class OandaNNet:
    def __init__(self, learning_rate=0.01, rate_size=32, position_size=3):
        self.output_size = ACTION_SIZE
        self.input_rate_size = rate_size
        self.input_position_size = position_size

        rates_input = Input(shape=(self.input_rate_size, self.input_rate_size, 1), name='rates_input')
        rate = Conv2D(64, kernel_size=(3, 3))(rates_input)
        rate = Activation('relu')(rate)
        rate = Conv2D(64, kernel_size=(3, 3))(rate)
        rate = Activation('relu')(rate)
        rate = MaxPooling2D(pool_size=(2, 2))(rate)
        rate = Dropout(0.25)(rate)
        rate = Dense(64, activation='relu')(rate)
        rate = Flatten()(rate)

        position_input = Input(shape=(self.input_position_size,), name='position_input')
        position = Dense(64)(position_input)

        main_input = Concatenate()([rate, position])
        main_input = Dense(128)(main_input)

        hdn = Dense(32, activation='relu')(main_input)

        v = Dense(32, activation='relu')(hdn)
        v = Dense(1)(v)

        adv = Dense(32, activation='relu')(hdn)
        adv = Dense(self.output_size)(adv)

        model = Concatenate()([v, adv])
        model = Lambda(lambda a: k.expand_dims(a[:, 0], -1) + a[:, 1:] - k.mean(a[:, 1:], axis=1, keepdims=True),
                       output_shape=(self.output_size,))(model)

        self._model = Model(inputs=[rates_input, position_input],
                            outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def input_data_format(self, lst):
        rates = []
        positions = []
        for s in lst:
            rate = np.reshape(s.rates.map, (self.input_rate_size, self.input_rate_size, 1))
            rates.append(rate)

            positions.append(s.position)
        return rates, positions

    def predict(self, x):
        rates, positions = self.input_data_format(x)
        return self._model.predict([rates, positions])

    def train_on_batch(self, x, y):
        rates, positions = self.input_data_format(x)
        y = np.reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch([rates, positions], y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()
