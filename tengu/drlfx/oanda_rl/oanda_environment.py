import copy
import math
from numpy import inf

import gym
import gym.spaces

from tengu.backtest.portfolio import Portfolio, LONG, SHORT

from logging import getLogger

from tengu.drlfx.modules.rate_list import RateSeries

logger = getLogger(__name__)

ERROR_REWARD = -1
NO_TRADE_REWARD = -1


def logformat(**kwargs):
    return kwargs


class OandaEnv(gym.Env):

    def __init__(self, env_name, rate_list, *, test_size=60 * 24 * 5, spread=0.018, err_handle_f=True):
        self.env_name = env_name
        self.portfolio = Portfolio(spread=spread, deposit=10000)
        self.rate_llist = RateSeries(rate_list, test_size=test_size)

        self.total_reward = 0
        self.done = False

        self.action_space = gym.spaces.Discrete(4)  # 取れる行動の数 0:何もしない 1:long open 2 sell open 3:close
        """
        self.observation_space = gym.spaces.Dict(
            {
                'rates': gym.spaces.Box(low=--inf, high=inf, shape=(self.rate_llist.state_size,)),  # 直近一時間のデータ
                'position': gym.spaces.Box(low=0, high=200, shape=(2,)),  # positionの状態 [long,short]
            }
        )
        """
        self.observation_space = gym.spaces.Box(-inf, inf, shape=(self.rate_llist.state_size + 4,))

        self.reward_range = [-1., 1.]

        self.err_handle_f = err_handle_f

        self.reset()

    def step(self, action):
        done = False
        reward = 0
        err_msg = None
        err_flag = False

        logger.debug(logformat(action="step", envname=self.env_name, step=self.rate_llist.index,
                               action_num=action
                               ))
        try:
            if action == 0:
                if self.portfolio.has_deals():
                    reward = 0.0001
                else:
                    reward = -0.0001

            elif action == 1:  # deal
                done = self.open_position(self.rate_llist.index, LONG)
            elif action == 2:  # deal
                done = self.open_position(self.rate_llist.index, SHORT)
            else:  # close
                reward = self.close_position(self.rate_llist.index)
        except RuntimeError as e:
            if self.err_handle_f:
                reward = -0.1
            else:
                err_msg = str(e)
                err_flag = True
                done = True
                reward = -100

        # 次の時間に進む
        if not err_flag and not done:
            done, _ = self.rate_llist.next()
            current_rate = self.rate_llist.current_rate
            profit = self.portfolio.pl_rate(current_rate)

            if profit < -1:
                # loss cut
                logger.info(logformat(action="loss cut", envname=self.env_name))
                reward = reward + self.close_position(current_rate)
                done = True
            if profit > 1:
                # profit lock
                logger.info(logformat(action="loss cut", envname=self.env_name))
                reward = reward + self.close_position(current_rate)
                done = True
            if self.portfolio.has_deals() and self.portfolio.current_balance(current_rate) < 0:
                # 強制ロスカット,罰則
                done = True

        # 取引が終了
        if not err_flag and done:
            # 一回も取引していない場合は罰則
            if len(self.portfolio.trading) == 0:
                reward = reward + NO_TRADE_REWARD

            # ポジション持ったまま終了 -> 決済して終わり
            if self.portfolio.has_deals():
                reward = reward + self.close_position(self.rate_llist.index)

            self.total_reward += reward
            print(logformat(action="finish trading", envname=self.env_name, step=self.rate_llist.index,
                                  trading_num=len(self.portfolio.trading), total_reward=self.total_reward,
                                  last_balance=self.portfolio.balance
                                  ))
        elif err_flag:
            self.total_reward = reward
            logger.warning(logformat(
                action="err_trade", envname=self.env_name, err_msg=err_msg, step=self.rate_llist.index,
                last_balance=self.portfolio.balance)
            )
        else:
            self.total_reward += reward

        self.done = done
        observe = self.observe()
        reward = reward
        info = {}
        logger.debug(logformat(action="stepend", envname=self.env_name, reward=reward, observe=observe
                               ))

        return observe, reward, self.done, info

    def reset(self):
        self.portfolio.reset(deposit=10000)
        self.rate_llist.reset()
        self.done = False
        self.total_reward = 0
        logger.debug(logformat(action="environment reset", envname=self.env_name, rate_index=self.rate_llist.start_pos))

        return self.observe()

    def render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def observe(self):
        # BLANK_STATUS = [0,0,0]
        # if self.is_done():
        #     return BLANK_STATUS

        rates = copy.copy(self.rate_llist.state)
        if self.portfolio.deals is None:
            position = [0, 0, 0, 0]
        elif self.portfolio.deals.position_type == LONG:
            position = [1, 0, self.portfolio.position_rate(), 0]
        elif self.portfolio.deals.position_type == SHORT:
            position = [0, 1, 0, self.portfolio.position_rate()]
        else:
            raise RuntimeError("そんなポジション認められてませーん")
        """
        observation = {
            'rates': rates,
            'position': position
        }
        """
        rates.extend(position)
        observation = rates
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

        current_rate = self.rate_llist.current_rate
        amount = calc_deal_amount(current_rate, self.portfolio.balance)
        position_tbl = {
            LONG: "LONG",
            SHORT: "SHORT"
        }
        logger.debug(
            logformat(action="open position", envname=self.env_name, step=step, position_type=position_tbl[position],
                      rate=current_rate, amount=amount
                      ))
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
        current_rate = self.rate_llist.current_rate
        profit = self.portfolio.current_profit(current_rate)

        self.portfolio.close_deal(step_num, current_rate, self.portfolio.deals.amount)
        logger.debug(logformat(action="close deal",
                               envname=self.env_name, step=step_num, position_rate=position_rate,
                               close_rate=current_rate, step_reward=profit, balance=self.portfolio.balance))
        return profit


if __name__ == '__main__':
    env = OandaEnv("name", rate_list=[i for i in range(1000)], test_size=100)
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
