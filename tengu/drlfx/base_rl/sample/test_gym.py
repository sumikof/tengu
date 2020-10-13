import os
import gym
import numpy as np

from tengu.drlfx.base_rl.test_abc import TestABC
from logging import getLogger

logger = getLogger(__name__)


class TestCartPole(TestABC):
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.num_actions = self.env.action_space.n  # 取れる行動の数
        num = self.env.observation_space.shape[0]
        self.num_status = num  # 状態を表す変数の数
        self.shape_status = [1, self.num_status]
        self.complete_episodes = 0

    @property
    def blank_status(self):
        return np.zeros(self.num_status)

    @property
    def mask(self):
        return [True, True]

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action, environment):
        next_state, reward, done, info = self.env.step(action)

        if done:
            #print("done is step : " + str(environment.step))
            next_state = self.blank_status

            # 報酬の設定
            if environment.step < 195:
                # 失敗
                reward = -1
                self.complete_episodes = 0
            else:
                # 成功
                reward = 1
                self.complete_episodes += 1
        else:
            # 終了時以外は報酬なし
            reward = 0

        return next_state, reward, done, info

    def is_finish(self):
        return self.complete_episodes > 10

    def check_status_is_done(self, state):
        return (state == self.blank_status).all()


if __name__ == '__main__':
    from logging import basicConfig, INFO

    basicConfig(level=INFO)

    test = TestCartPole()

    from tengu.drlfx.base_rl.nnet_builder.nnet_builder import NNetBuilder
    env = NNetBuilder(test, "DDQN").build_environment()
    env.run()
