from rl.core import Processor


class OandaProcessor(Processor):
    def process_reward(self, reward):
        return reward
