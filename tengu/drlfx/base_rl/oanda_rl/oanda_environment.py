import math
from numpy import inf

import gym
import gym.spaces

from tengu.backtest.portfolio import Portfolio, LONG, SHORT

from logging import getLogger

from tengu.drlfx.base_rl.modules.rate_list import RateList

logger = getLogger(__name__)

ERROR_REWARD = -1
NO_TRADE_REWARD = -1


class OandaEnv(gym.Env):

    def __init__(self, rate_list, *, rate_size=1, test_size=60 * 24 * 5, spread=0.018):
        self.portfolio = Portfolio(spread=spread, deposit=10000)
        self.exchanger = RateList(rate_list, state_size=rate_size, test_size=test_size)

        self.total_reward = 0
        self.done = False

        self.action_space = gym.spaces.Discrete(4)  # 取れる行動の数 0:何もしない 1:long open 2 sell open 3:close
        self.observation_space = gym.spaces.Box(-inf, inf, shape=(3,))
        """
        self.observation_space = gym.spaces.Dict(
            {
                'rates': gym.spaces.Box(low=0, high=200, shape=(rate_size,)),  # 直近一時間のデータ
                'position': gym.spaces.Box(low=0, high=200, shape=(2,)),  # positionの状態 [long,short]
            }
        )
        """
        self.reward_range = [-1., 1.]

        self.reset()

    def step(self, action):
        done = False
        reward = 0
        try:
            if action == 0:
                pass
            elif action == 1:  # deal
                done = self.open_position(self.exchanger.index, LONG)
            elif action == 2:  # deal
                done = self.open_position(self.exchanger.index, SHORT)
            else:  # close
                reward = self.close_position(self.exchanger.index)
        except RuntimeError:
            done = True
            reward = -1

        #
        if not done:
            done, _ = self.exchanger.next()
            current_rate = self.exchanger.rate[-1]
            profit = self.current_profit(current_rate)

            if profit < -1:
                # loss cut
                reward = self.close_position(current_rate)
                logger.info("loss cut")
                done = True
            if profit > 1:
                # profit lock
                reward = self.close_position(current_rate)
                logger.info("profit lock")
                done = True
            if self.portfolio.has_deals() and self.portfolio.current_balance(current_rate) < 0:
                # 強制ロスカット,罰則
                done = True

        #
        if done:
            if len(self.portfolio.trading) == 0:
                # 一回も取引していない場合は罰則
                reward = NO_TRADE_REWARD
            if self.portfolio.has_deals():
                # ポジション持ったまま終了 -> 決済して終わり
                reward = self.close_position(self.exchanger.index)
                logger.info(
                    "finish close deal,rate {} ,self.step_reward {}".format(self.exchanger.rate[-1], reward))

            logger.info("step index {} ,trading num {} ,finish total_reward {} last balance{}".format(
                self.exchanger.index, len(self.portfolio.trading), self.total_reward, self.portfolio.balance))

        self.done = done
        observe = self.observe()
        reward = reward
        info = {}
        return observe, reward, self.done, info

    def reset(self):
        self.portfolio.reset(deposit=10000)
        self.exchanger.reset()
        self.done = False

        return self.observe()

    def render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def observe(self):
        # BLANK_STATUS = [0,0,0]
        # if self.is_done():
        #     return BLANK_STATUS

        rates = self.exchanger.copy_rate()
        if self.portfolio.deals is None:
            position = [0, 0]
        elif self.portfolio.deals.position_type == LONG:
            position = [self.portfolio.position_rate(), 0]
        elif self.portfolio.deals.position_type == SHORT:
            position = [0, self.portfolio.position_rate()]
        else:
            raise RuntimeError("そんなポジション認められてませーん")
        rates.extend(position)
        observation = rates
        """
        observation = {
            'rates': rates,
            'position': position
        }
        """
        return observation

    def is_done(self):
        return self.done

    def open_position(self, step, position):
        if self.portfolio.has_deals():
            # すでにpositionある
            raise RuntimeError("position あるのに{}建てしてるよ".format(position))

        done = False

        def calc_deal_amount(rate, balance):
            """ 取引量を算出 """
            leverage = 100
            lot = 1000
            margin = balance * leverage
            if not rate == 0:
                _amount = math.floor(margin / rate / lot) * lot
            else:
                _amount = 1
            return _amount

        current_rate = self.exchanger.rate[-1]
        amount = calc_deal_amount(current_rate, self.portfolio.balance)
        logger.info("step {},open long deal,rate {:.3f}, amount {}".format(step, current_rate, amount))
        if amount < 1:
            # 残高不足、もう取引できない
            done = True
        else:
            self.portfolio.deal(step, position, current_rate, amount)
        return done

    def close_position(self, step_num):
        if not self.portfolio.has_deals():
            # ポジションない
            raise RuntimeError("position ないのに決済してるよ")

        # ポジションを決済
        position_rate = self.portfolio.deals.rate
        current_rate = self.exchanger.rate[-1]
        profit = self.current_profit(current_rate)

        logger.info("step is {} position rate {:.3f} ,close rate {:.3f} ,self.step_reward {:.3f}".format(
            step_num, position_rate, current_rate, profit))
        self.portfolio.close_deal(step_num, current_rate, self.portfolio.deals.amount)
        return profit

    def current_profit(self, current_rate):
        if not self.portfolio.has_deals():
            return 0
        position_rate = self.portfolio.deals.rate
        profit = (current_rate / position_rate - 1) * self.portfolio.deals.position_type  # * 100
        return profit


if __name__ == '__main__':
    env = OandaEnv(rate_list=[i for i in range(1000)], rate_size=1, test_size=100)
    print(env.action_space.n)
    print(env.observation_space.shape)
    obs = env.reset()
    print(obs)
    obs = env.step(0)
    print(obs)
    obs = env.step(1)
    print(obs)
    obs = env.step(0)
    print(obs)
    obs = env.step(0)
    print(obs)
    obs = env.step(3)
    print(obs)
    obs = env.step(0)
    print(obs)
    while not obs[2]:
        obs = env.step(0)
        print(obs)
    print(env.portfolio.trading)
