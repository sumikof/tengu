import random
import copy
import math
import numpy as np
from tengu.backtest.portfolio import Portfolio
from tengu.backtest.portfolio import LONG, SHORT

from collections import namedtuple

from tengu.drlfx.base_rl.test_abc import TestABC
from logging import getLogger, DEBUG

logger = getLogger(__name__)

StepState = namedtuple('stepstate', ('rates', 'position', 'current_rate', 'mask'))

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


ERROR_REWARD = -1
NO_TRADE_REWARD = -1


class TestOanda(TestABC):
    position_mask = np.array([1, 0, 0, 1])
    non_position_mask = np.array([1, 1, 1, 0])
    weight_file_name = 'test_oanda.hdf5'

    def __init__(self, rate_list, batch_size, rate_size, spread=0.018, save_weights=False):

        self.num_actions = ACTION_SIZE  # 取れる行動の数 0:何もしない 1:long open 2 sell open 3:close
        self.rate_size = rate_size
        self.batch_size = batch_size
        self.rate_list = rate_list
        self.portfolio = Portfolio(spread=spread)

        self.batch = None
        self.complete_episodes = 0
        self.index = 0
        self.end_index = 0
        self._step_reward = 0
        self.total_reward = 0

        self.rate_map = RateMap(self.rate_size)
        self._save_weights = save_weights
        self.reset()

    @property
    def save_weights(self):
        return self._save_weights

    @save_weights.setter
    def save_weights(self, bool):
        self._save_weights = bool

    @property
    def step_reward(self):
        return self._step_reward

    @step_reward.setter
    def step_reward(self, reward):
        self._step_reward = reward
        self.total_reward += reward

    @property
    def mask(self):
        if self.portfolio.has_deals():
            return self.position_mask
        else:
            return self.non_position_mask

    @property
    def blank_status(self):
        return None

    @property
    def current_rate(self):
        return self.batch[self.end_index]

    @property
    def profit_reward(self):
        return self._profit() * 10

    @property
    def position_rate(self):
        if self.portfolio.has_deals():
            position_rate = self.portfolio.position_rate()
        else:
            position_rate = 0
        return position_rate

    @property
    def state(self):

        if self.portfolio.deals is None:
            position = np.array([0, 0])
        elif self.portfolio.deals.position_type == LONG:
            position = np.array([1, 0])
        elif self.portfolio.deals.position_type == SHORT:
            position = np.array([0, 1])
        else:
            print("そんなポジション認められてませーん")
            raise NotImplementedError
        position = position * self.portfolio.current_profit(self.current_rate)

        state = StepState(np.array(copy.copy(self.rate_map).map),
                          copy.copy(position),
                          copy.copy(self.current_rate),
                          copy.copy(self.mask))

        return state

    def reset(self):
        self.portfolio.reset()
        self.portfolio.deposit(10000)
        self.batch = self._minibatch()
        self.index = 0
        self.end_index = self.index + self.rate_size - 1
        self.total_reward = 0

        self.rate_map.reset()
        self.rate_map.push(self._get_rates())
        for i in range(self.rate_size):
            self._add_index()
        return self.state

    def _minibatch(self):
        random_index = random.randint(0, len(self.rate_list) - self.batch_size)
        return self.rate_list[random_index:random_index + self.batch_size]

    def _add_index(self):
        self.index += 1
        self.end_index += 1
        done = self.end_index >= self.batch_size - 1
        self.rate_map.push(self._get_rates())
        return done

    def _get_rates(self):
        return self.batch[self.index:self.index + self.rate_size]

    def _losscut(self):
        """強制ロスカット"""
        self.portfolio.current_profit(self.current_rate)
        return False

    def _profit(self):
        if self.portfolio.has_deals():
            position_rate = self.portfolio.position_rate()
            current_rate = self.current_rate
            profit = (current_rate / position_rate - 1) * 100 * self.portfolio.deals.position_type
        else:
            profit = 0

        return profit

    def step(self, action, env):
        self.step_reward = 0
        done = False
        if action == 0:
            pass
        elif action == 1:  # deal
            if self.portfolio.has_deals():
                # すでにpositionある
                logger.error("position あるのに買建してるよ")
                raise NotImplementedError

            amount = self.calc_deal_amount()
            logger.info("step is {} open long deal,rate {:.3f}, amount {}".format(env.step, self.current_rate, amount))
            if amount < 1:
                done = True
            else:
                self.portfolio.deal(env.step, LONG, self.current_rate, amount)

        elif action == 2:  # deal
            if self.portfolio.has_deals():
                # すでにpositionある
                logger.error("position あるのに売建してるよ")
                raise NotImplementedError

            amount = self.calc_deal_amount()
            logger.info("step is {} open short deal,rate {:.3f},amount {}".format(env.step, self.current_rate, amount))
            if amount < 1:
                done = True
            else:
                self.portfolio.deal(env.step, SHORT, self.current_rate, amount)

        else:  # close
            if not self.portfolio.has_deals():
                # ポジションない
                logger.error("position ないのに決済してるよ")
                raise NotImplementedError

            # ポジションを決済
            position_rate = self.position_rate
            current_rate = self.current_rate
            self.step_reward = self.profit_reward

            logger.info("step is {} position rate {:.3f} ,close rate {:.3f} ,self.step_reward {:.3f}".format(
                env.step, position_rate, current_rate, self.step_reward))
            self.portfolio.close_deal(env.step, self.current_rate, self.portfolio.deals.amount)

        if not done:
            done = self._add_index()
            if self.portfolio.has_deals() and self.portfolio.current_balance(self.current_rate) < 0:
                # 強制ロスカット,罰則
                done = True

        if done:
            if len(self.portfolio.trading) == 0:
                # 一回も取引していない場合は罰則
                self.step_reward = NO_TRADE_REWARD

            if self.portfolio.has_deals():
                # ポジション持ったまま終了 -> 決済して終わり
                self.step_reward = self.profit_reward

                logger.info(
                    "finish close deal,rate {} ,self.step_reward {}".format(self.current_rate, self.step_reward))
                self.portfolio.close_deal(env.step, self.current_rate, self.portfolio.deals.amount)

            next_state = self.blank_status
            logger.info("step index {} ,trading num {} ,finish total_reward {} last balance{}".format(
                self.index, len(self.portfolio.trading), self.total_reward, self.portfolio.balance))
        else:
            next_state = self.state

        info = None

        return next_state, self.step_reward, done, info

    def is_finish(self):
        return self.complete_episodes > 10

    def check_status_is_done(self, state):
        return state is None

    def calc_deal_amount(self):
        rate = self.current_rate
        balance = self.portfolio.balance
        leverage = 100
        lot = 1000
        margin = balance * leverage
        amount = math.floor(margin / rate / lot) * lot

        return amount

    def __copy__(self):
        return TestOanda(rate_list=self.rate_list, batch_size=self.batch_size, rate_size=self.rate_size,
                         spread=self.portfolio.spread,
                         save_weights=self.save_weights)


def test_execute():
    from tengu.oanda_action.oanda_dataframe import oanda_dataframe
    from tengu.drlfx.base_rl.agent import AgentDDQN
    from tengu.drlfx.base_rl.brain import BrainDDQN
    from tengu.drlfx.oanda_nnet import OandaNNet

    df_org = oanda_dataframe('../../USD_JPY_M1.csv')
    rate_size = 6
    test = TestOanda(df_org['close'].values, (60), rate_size)
    test.save_weights = True
    logger.setLevel(DEBUG)
    eta = 0.0001  # 学習係数

    brain = BrainDDQN(test,
                      main_network=OandaNNet(learning_rate=eta, rate_size=rate_size),
                      target_network=OandaNNet(learning_rate=eta, rate_size=rate_size))
    agent = AgentDDQN(brain)

    from tengu.drlfx.base_rl.environment import EnvironmentDDQN
    env = EnvironmentDDQN(test, agent, num_episodes=500, max_steps=0)

    env.run()


if __name__ == '__main__':
    test_execute()
