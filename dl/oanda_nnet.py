from keras import Model, Input
from keras.layers import Dense, Concatenate, Lambda
from keras.optimizers import Adam
import keras.backend as k
from numpy import reshape
from dl.base_rl.loss_function import huberloss
from dl.test_oanda import RATE_DATA_SIZE


def input_data_format(lst):
    rates = []
    current_rate = []
    current_profit = []
    has_deals = []
    for s in lst:
        rates.append(s.rates)
        current_rate.append(s.current_rate)
        current_profit.append(s.current_profit)
        has_deals.append(s.has_deals)
    return rates, current_rate, current_profit, has_deals


class OandaNNet:
    def __init__(self, learning_rate=0.01, hidden_size=10):
        self.output_size = 3

        rates_input = Input(shape=(RATE_DATA_SIZE,), name='rates_input')
        rate = Dense(32)(rates_input)

        current_rate_input = Input(shape=(1,), name='current_rate_input')
        current_rate = Dense(32)(current_rate_input)

        current_profit_input = Input(shape=(1,), name='current_profit_input')
        current_profit = Dense(32)(current_profit_input)

        has_deals_input = Input(shape=(1,), name='has_deals_input')
        #has_deals = Dense(32)(has_deals_input)

        main_input = Concatenate()([rate, current_rate, current_profit, has_deals_input])

        hdn = Dense(hidden_size, activation='relu')(main_input)

        v = Dense(hidden_size, activation='relu')(hdn)
        v = Dense(1)(v)

        adv = Dense(hidden_size, activation='relu')(hdn)
        adv = Dense(self.output_size)(adv)

        model = Concatenate()([v, adv])
        model = Lambda(lambda a: k.expand_dims(a[:, 0], -1) + a[:, 1:] - k.mean(a[:, 1:], axis=1, keepdims=True),
                       output_shape=(self.output_size,))(model)

        self._model = Model(inputs=[rates_input, current_rate_input, current_profit_input, has_deals_input],
                            outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def predict(self, x):
        rates, current_rate, current_profit, has_deals = input_data_format(x)
        return self._model.predict([rates, current_rate, current_profit, has_deals])

    def train_on_batch(self, x, y):
        rates, current_rate, current_profit, has_deals = input_data_format(x)
        y = reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch([rates, current_rate, current_profit, has_deals], y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()
