from random import randint, choices
import numpy as np
from backtest.portfolio import Portfolio
from common.util import oanda_dataframe
from backtest.portfolio import LONG

from collections import namedtuple

from dl.test_abc import TestABC

StepState = namedtuple('stepstate', ('rates', 'current_rate', 'current_profit', 'has_deals'))


class TestOanda(TestABC):
    position_mask = np.array([True, False, True])
    non_position_mask = np.array([True, True, False])

    def __init__(self, lst):

        self.num_actions = 3  # 取れる行動の数 0:何もしない 1:deal 2:close
        self.data_size = 32
        self.num_status = 64  # 状態を表す変数の数
        self.batch_size = 1000
        self.lst = lst
        self.complete_episodes = 0
        self.portfolio = None
        self.__mask = [True, True, False]
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
        random_index = randint(0, len(self.lst) - self.batch_size)
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
                raise NotImplementedError

        if not done:
            done = self._add_index()

        next_state = self._getstate()

        info = None

        return next_state, reward, done, info

    def is_finish(self):
        return self.complete_episodes > 10


if __name__ == '__main__':
    df_org = oanda_dataframe('../USD_JPY_M1.csv')
    test = TestOanda(df_org['close'].values)


    class TestData:
        pass


    done = False
    step = 0
    total_reward = 0
    while not done:
        print("execute step {0} test step {1} end {2}".format(step, test.index, test.endindex))
        env = TestData()
        env.step = step
        action = choices(np.array([0, 1, 2]) * test.mask)[0]
        state, reward, done, _ = test.step(action, env)
        if reward != 0:
            print("reward :{0:.3f}".format(reward))
            total_reward += reward
        step += 1
    print("total reward :{0:.3f}".format(total_reward))
