from keras import Model, Input
from keras.layers import Dense, Concatenate
from keras.optimizers import Adam

from dl.ddqn.loss_function import huberloss


class OandaNNet:
    def __init__(self, learning_rate=0.01, hidden_size=10):
        self.output_size = 3

        rates = Input(shape=(32,), name='rates')
        current_rate = Input(shape=(1,))
        current_profit = Input(shape=(1,))
        has_deals = Input(shape=(1,))
        main_input = Concatenate()([rates, current_rate, current_profit, has_deals])

        model = Dense(hidden_size, activation='relu')(main_input)
        model = Dense(hidden_size, activation='relu')(model)
        model = Dense(self.output_size, activation='linear')(model)

        self._model = Model(inputs=[rates, current_rate, current_profit, has_deals], outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def predict(self, input):
        rates, current_rate, current_profit, has_deals = self.input_data_format(input)
        return self._model.predict([rates, current_rate, current_profit, has_deals])

    def train_on_batch(self, x, y):
        rates, current_rate, current_profit, has_deals = self.input_data_format(x)
        y = np.reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch([rates, current_rate, current_profit, has_deals], y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()

    def input_data_format(self, lst):
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


import random
import numpy as np
from backtest.portfolio import Portfolio
from common.util import oanda_dataframe
from backtest.portfolio import LONG

from collections import namedtuple

from dl.ddqn.environment import EnvironmentDDQN
from dl.test_abc import TestABC

StepState = namedtuple('stepstate', ('rates', 'current_rate', 'current_profit', 'has_deals'))


class TestOanda(TestABC):
    position_mask = np.array([True, False, True])
    non_position_mask = np.array([True, True, False])

    def __init__(self, lst):

        self.num_actions = 3 # 取れる行動の数 0:何もしない 1:deal 2:close
        self.data_size = 32
        self.num_status = 64  # 状態を表す変数の数
        self.batch_size = 1000
        self.lst = lst
        self.complete_episodes = 0
        self.portfolio = None
        self.reset()

    @property
    def mask(self):
        if self.portfolio.has_deals():
            return self.position_mask
        else:
            return self.non_position_mask

    def _getstate(self):
        state = StepState(self.batch[self.index:self.index + self.data_size],
                          self._get_rate(),
                          self.portfolio.current_profit(self._get_rate()),
                          self.portfolio.has_deals())
        return state

    def _minibatch(self):
        random_index = random.randint(0, len(self.lst) - self.batch_size)
        return self.lst[random_index:random_index + self.batch_size]

    def reset(self):
        self.portfolio = Portfolio(spread=0.018)
        self.portfolio.deposit(10000)
        self.batch = self._minibatch()
        self.index = 0
        self.endindex = self.index + self.data_size - 1
        return self._getstate()

    def _add_index(self):
        self.index += 1
        self.endindex += 1
        done = self.endindex >= self.batch_size - 1
        return done

    def _get_rate(self):
        return self.batch[self.endindex]

    def _losscut(self):
        """強制ロスカット"""
        self.portfolio.current_profit(self._get_rate())
        return False

    def _profit(self):
        position_rate = self.portfolio.position_rate()
        current_rate = self._get_rate()
        return (current_rate / position_rate - 1) * 100

    def step(self, action, environment):
        reward = 0
        done = False
        if action == 0:
            pass
        elif action == 1:  # deal
            if self.portfolio.has_deals():
                print("position あるのに買ってるよ")
                print("mask : " + str(self.mask))
                print(env.env.portfolio.has_deals())
                # すでにpositionある
                raise NotImplementedError
            else:
                self.portfolio.deal(environment.step, LONG, self._get_rate(), 100)
        else:  # close
            if self.portfolio.has_deals():
                # すでにpositionある
                reward = self._profit()
                self.portfolio.close_deal(environment.step, self._get_rate(), 100)
            else:
                print("position ないのに決済してるよ")
                print("mask : " + str(self.mask))
                print(env.env.portfolio.has_deals())
                raise NotImplementedError

        if not done:
            done = self._add_index()

        if done:
            next_state = self.blank_status()
        else:
            next_state = self._getstate()

        info = None

        return next_state, reward, done, info

    def is_finish(self):
        return self.complete_episodes > 10

    def check_status_is_done(self, state):
        return state is None

    def blank_status(self):
        return None





if __name__ == '__main__':
    df_org = oanda_dataframe('../USD_JPY_M1.csv')
    test = TestOanda(df_org['close'].values)

    ETA = 0.0001  # 学習係数
    learning_rate = ETA
    hidden_size = 32

    from dl.ddqn.agent import AgentDDQN
    from dl.ddqn.brain import BrainDDQN

    brain = BrainDDQN(test,
                      main_network=OandaNNet(learning_rate, hidden_size),
                      target_network=OandaNNet(learning_rate,  hidden_size))
    agent = AgentDDQN(brain)
    env = EnvironmentDDQN(test, agent, max_steps=0)
    env.run()
