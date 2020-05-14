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


    def step(self,action,environtment):
        next_state, reward, done, info = self.env.step(action)

        next_state = np.reshape(next_state, [1, self.num_status])

        if done:
            print("done is step : " + str(environtment.step))
            next_state = np.zeros([1, self.num_status])

            # 報酬の設定
            if environtment.step < 195:
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

if __name__ =='__main__':

    test = TestCartPole()
    print(test.num_actions)
    print(test.num_status)
    print(test.reset())

    from dl.ddqn.environment import EnvironmentDDQN
    env = EnvironmentDDQN(test)

    env.run()


