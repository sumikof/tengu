import random
import copy
import numpy as np
from backtest.portfolio import Portfolio
from backtest.portfolio import LONG, SHORT

from collections import namedtuple

from dl.base_rl.test_abc import TestABC

StepState = namedtuple('stepstate', ('rates', 'position'))

ACTION_SIZE = 4


class RateMap:
    def __init__(self, size):
        self.size = size
        self._map = np.empty((size, size))

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, m):
        self._map = m

    def push(self, rates):
        if len(rates) != self.size:
            raise RuntimeError("format error len != {} ,act = {}".format(self.size, len(rates)))
        self.map = np.append(self.map, [rates], axis=0)
        if len(self.map) > self.size:
            self.map = np.delete(self.map, 0, 0)

    def reset(self):
        self.map = np.empty((self.size, self.size))

    def __copy__(self):
        t = RateMap(self.size)
        t.map = copy.deepcopy(self.map)
        return t


class TestOanda(TestABC):
    position_mask = np.array([True, False, False, True])
    non_position_mask = np.array([True, True, True, False])

    def __init__(self, lst, batch_size, rate_size):

        self.num_actions = ACTION_SIZE  # 取れる行動の数 0:何もしない 1:long open 2 sell open 3:close
        self.data_size = rate_size
        self.batch_size = batch_size
        self.lst = lst
        self.complete_episodes = 0
        self.portfolio = Portfolio(spread=0.018)

        self.batch = None
        self.index = 0
        self.end_index = 0
        self.total_reward = 0

        self.rate_map = RateMap(self.data_size)
        self.reset()

    def reset(self):
        self.portfolio.reset()
        self.portfolio.deposit(10000)
        self.batch = self._minibatch()
        self.index = 0
        self.end_index = self.index + self.data_size - 1
        self.total_reward = 0

        self.rate_map.reset()
        self.rate_map.push(self._get_rates())
        for i in range(self.data_size):
            self._add_index()
        return self._getstate()

    @property
    def mask(self):
        if self.portfolio.has_deals():
            return self.position_mask
        else:
            return self.non_position_mask

    def _getstate(self):

        if self.portfolio.deals is None:
            position = np.array([0, 0])
        elif self.portfolio.deals.position_type == LONG:
            position = np.array([1, 0])
        elif self.portfolio.deals.position_type == SHORT:
            position = np.array([0, 1])
        else:
            print("そんなポジション認められてませーん")
            raise NotImplementedError
        position = position * self.portfolio.current_profit(self._get_current_rate())

        state = StepState(copy.copy(self.rate_map),
                          np.append(position,self._get_current_rate()))
        return state

    def _minibatch(self):
        random_index = random.randint(0, len(self.lst) - self.batch_size)
        return self.lst[random_index:random_index + self.batch_size]

    def _add_index(self):
        self.index += 1
        self.end_index += 1
        done = self.end_index >= self.batch_size - 1
        self.rate_map.push(self._get_rates())
        return done

    def _get_rates(self):
        return self.batch[self.index:self.index + self.data_size]

    def _get_current_rate(self):
        return self.batch[self.end_index]

    def _losscut(self):
        """強制ロスカット"""
        self.portfolio.current_profit(self._get_current_rate())
        return False

    def _profit(self):
        position_rate = self.portfolio.position_rate()
        current_rate = self._get_current_rate()
        return (current_rate / position_rate - 1) * 100

    def step(self, action, environment):
        reward = 0
        done = False
        if action == 0:
            pass
        elif action == 1:  # deal
            if self.portfolio.has_deals():
                print("position あるのに買建してるよ")
                print("mask : " + str(self.mask))
                print(self.portfolio.has_deals())
                # すでにpositionある
                raise NotImplementedError
            else:
                log("step is {} open long deal,rate {}".format(environment.step, self._get_current_rate()))
                rate = self._get_current_rate()
                balance = self.portfolio.balance
                leverage = 100
                lot = 1000
                margin = balance * leverage
                import math
                amount = math.floor(margin / rate / lot) * lot
                if amount < 1:
                    done = True
                else:
                    self.portfolio.deal(environment.step, LONG, self._get_current_rate(), amount)
                    # print(self.portfolio.trading[-1])
        elif action == 2:  # deal
            if self.portfolio.has_deals():
                print("position あるのに売建してるよ")
                print("mask : " + str(self.mask))
                print(self.portfolio.has_deals())
                # すでにpositionある
                raise NotImplementedError
            else:
                log("step is {} open short deal,rate {}".format(environment.step, self._get_current_rate()))
                rate = self._get_current_rate()
                balance = self.portfolio.balance
                leverage = 100
                lot = 1000
                margin = balance * leverage
                import math
                amount = math.floor(margin / rate / lot) * lot
                if amount < 1:
                    done = True
                else:
                    self.portfolio.deal(environment.step, SHORT, self._get_current_rate(), amount)

        else:  # close
            if self.portfolio.has_deals():
                # すでにpositionある
                reward = self._profit() * 10
                log("step is {} close deal,rate {} ,reward {}".format(environment.step, self._get_current_rate(),
                                                                        reward))
                self.portfolio.close_deal(environment.step, self._get_current_rate(), self.portfolio.deals.amount)

            else:
                print("position ないのに決済してるよ")
                print("mask : " + str(self.mask))
                print(self.portfolio.has_deals())
                raise NotImplementedError

        if not done:
            done = self._add_index()
            if self.portfolio.has_deals() and self.portfolio.current_balance(self._get_current_rate()) < 0:
                # 強制ロスカット
                done = True
                reward = -10

        self.total_reward += reward

        if done:
            if len(self.portfolio.trading) == 0:
                reward = -10
                self.total_reward += reward

            if self.portfolio.has_deals():
                # すでにpositionある
                reward = self._profit() * 10
                log("finish close deal,rate {} ,reward {}".format(self._get_current_rate(), reward))
                self.portfolio.close_deal(environment.step, self._get_current_rate(), self.portfolio.deals.amount)


            next_state = self.blank_status()
            print("step index {} ,trading num {} ,finish total_reward {} last balance".format(
                self.index, len(self.portfolio.trading), self.total_reward), self.portfolio.balance)
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

def log(msg):
    #print(msg)
    pass

def test_execute():
    from oanda_action.oanda_dataframe import oanda_dataframe
    df_org = oanda_dataframe('../USD_JPY_M1.csv')
    rate_size = 64
    test = TestOanda(df_org['close'].values, (60 * 24*5), rate_size)

    ETA = 0.0001  # 学習係数
    learning_rate = ETA
    hidden_size = 32

    from dl.base_rl.agent import AgentDDQN
    from dl.base_rl.brain import BrainDDQN
    from dl.oanda_nnet import OandaNNet
    brain = BrainDDQN(test,
                      main_network=OandaNNet(learning_rate, hidden_size, rate_size=rate_size),
                      target_network=OandaNNet(learning_rate, hidden_size, rate_size=rate_size))
    agent = AgentDDQN(brain)

    from dl.base_rl.environment import EnvironmentDDQN
    env = EnvironmentDDQN(test, agent, num_episodes=500, max_steps=0)
    env.run()


if __name__ == '__main__':
    test_execute()
