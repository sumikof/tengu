from rl.core import Processor
from logging import getLogger

logger = getLogger(__name__)


class OandaProcessor(Processor):

    def process_observation(self, observation):
        logger.debug("OandaProcessor State: {}".format(observation))
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        logger.debug("OandaProcessor Action :{}".format(action))
        return action
