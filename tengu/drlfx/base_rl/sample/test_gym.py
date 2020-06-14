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
        self._save_weights = False
        self.weight_file_name = 'test_cart_pole.hdf5'

    @property
    def save_weights(self):
        return self._save_weights

    @save_weights.setter
    def save_weights(self, is_save):
        self._save_weights = is_save

    @property
    def mask(self):
        return [True, True]

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action, environment):
        next_state, reward, done, info = self.env.step(action)

        if done:
            print("done is step : " + str(environment.step))
            next_state = np.zeros(self.num_status)

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
        return (state == np.zeros(self.num_status)).all()


if __name__ == '__main__':
    from logging import basicConfig, INFO

    basicConfig(level=INFO)

    test = TestCartPole()
    test.save_weights = False
    test.reset()

    from tengu.drlfx.base_rl.environment import EnvironmentDDQN

    ETA = 0.0001  # 学習係数
    learning_rate = ETA
    hidden_size = 32

    from tengu.drlfx.base_rl.agent import AgentDDQN
    from tengu.drlfx.base_rl.brain import BrainDDQN
    from tengu.drlfx.base_rl.sample.simple_nnet import SimpleNNet

    main_network = SimpleNNet(learning_rate, test.num_status, test.num_actions, hidden_size)
    target_network = SimpleNNet(learning_rate, test.num_status, test.num_actions, hidden_size)

    base_epsilon = 0.5

    load_weight = False
    if load_weight and os.path.isfile(test.weight_file_name):
        logger.debug("load weight_file: {}".format(test.weight_file_name))
        main_network.load_weights(test.weight_file_name)
        target_network.load_weights(test.weight_file_name)
        main_network.model.summary(print_fn=logger.debug)
        base_epsilon = 0.001

    memory_capacity = 10000
    per_alpha = 0.6
    from tengu.drlfx.base_rl import experience_memory

    memory = experience_memory.factory(memory_type=experience_memory.type.PERRankBaseMemory,
                                       memory_capacity=memory_capacity, per_alpha=per_alpha)

    brain = BrainDDQN(task=test, memory=memory,
                      main_network=main_network,
                      target_network=target_network,
                      base_epsilon=base_epsilon
                      )

    agent = AgentDDQN(brain)
    env = EnvironmentDDQN(test, agent, max_steps=0)

    env.run()
