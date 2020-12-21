from tengu.drlfx.base_rl.agent.agent57 import ActorUser
from tengu.drlfx.base_rl.agent.policy import EpsilonGreedy

class EnvironmentGenerator:
    def create_env(self):
        raise NotImplementedError

class EnvironmentManager:
    def __init__(self):
        self.generator = EnvironmentGenerator()

    def set_generator(self,generator):
        self.generator = generator

    def create_env(self):
        return self.generator.create_env()


env_manager = EnvironmentManager()


class MyActor(ActorUser):
    @staticmethod
    def allocate(actor_index, actor_num):
        return "/device:CPU:0"

    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

    def fit(self, index, agent):
        env = env_manager.create_env()
        agent.fit(env, visualize=False, verbose=0)
        env.close()


class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.01)


class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)


