import rl.core


class CartPoleProcessor(rl.core.Processor):
    """
    https://github.com/openai/gym/wiki/CartPole-v0
    """

    def __init__(self, enable_reward_step=False):
        self.enable_reward_step = enable_reward_step
        self.step = 0

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        if not self.enable_reward_step:
            return observation, reward, done, info

        self.step += 1

        if done:
            if self.step > 195:
                reward = 1
            else:
                reward = -1
            self.step = 0
        else:
            reward = 0

        return observation, reward, done, info

    def get_keys_to_action(self):
        return {
            (ord('a'),): 0,
            (ord('d'),): 1,
        }
