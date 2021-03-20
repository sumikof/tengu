from tengu.drlfx.agent.agent57 import ActorUser
from tengu.drlfx.agent.policy import EpsilonGreedy


class MyActor(ActorUser):
    def __init__(self, generator):
        self.generator = generator

    @staticmethod
    def allocate(actor_index, actor_num):
        return "/device:CPU:0"

    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

    def fit(self, index, agent):
        env = self.generator.create_env(index)
        agent.fit(env, visualize=False, verbose=0)
        env.close()


class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.01)


class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)
