from rl.core import Processor


class OandaProcessor(Processor):

    def process_observation(self, observation):
        print("OandaProcessor State: {}".format(observation))
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        print("OandaProcessor Action :{}".format(action))
        return action