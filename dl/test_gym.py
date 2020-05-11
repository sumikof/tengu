import random
import gym
import numpy as np

EPISODE_SIZE = 10
class TestCartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.num_actions = self.env.action_space.n # 取れる行動の数
        self.num_status = self.env.observation_space.shape[0] # 状態を表す変数の数
        self.complete_episodes = 0

    def reset(self):

        return self.env.reset()


    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info



if __name__ =='__main__':

    test = TestCartPole()
    print(test.num_actions)
    print(test.num_status)
    print(test.reset())

    from dl.ddqn.environment import EnvironmentDDQN
    env = EnvironmentDDQN(test)

    env.run()


