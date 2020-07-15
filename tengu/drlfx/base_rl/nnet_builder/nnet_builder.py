from enum import Enum, auto

from tengu.drlfx.base_rl.experience_memory.ReplayMemory import ReplayMemory
from tengu.drlfx.base_rl.sample.dueling_network import DuelingNNet
from tengu.drlfx.base_rl.sample.simple_nnet import SimpleNNet


class BuilderArgument:
    def __init__(self):
        self.table = {
            'memory_type': ReplayMemory,
            'nnet_type': SimpleNNet,
            'batch_size': 32,
            'gamma': 0.99,
            'base_epsilon': 0.99,
            'multi_reward_size': 3,
            'num_episodes': 300,
            'max_steps': 200,
            'train_interval': 2,
            'learning_rate': 0.01,
            'hidden_size': 10,
            'memory_capacity': 10000,
            'per_alpha': 0.6
        }

    def get(self, key, default_value):
        return self.table.get(key, default_value)

    def __getitem__(self, item):
        return self.table[item]


class NNetBuilder:
    class RlType(Enum):
        DDQN = auto()

    def __init__(self, test, rl_type, args=None, *, environment=None, agent=None, brain=None, nnet=None, memory=None):
        self.args = args or BuilderArgument()

        self.test = test

        l_agent, l_brain, l_environment = self.rl_set(rl_type)

        self.environment = environment or l_environment
        self.agent = agent or l_agent
        self.brain = brain or l_brain
        self.nnet = nnet or self.args['nnet_type']
        self.memory = memory or self.args["memory_type"]

    def rl_set(self, rl_type):
        from tengu.drlfx.base_rl.brain import BrainDDQN
        from tengu.drlfx.base_rl.agent import AgentDDQN
        from tengu.drlfx.base_rl.environment import EnvironmentDDQN
        agent = AgentDDQN
        brain = BrainDDQN
        environment = EnvironmentDDQN
        return agent, brain, environment

    def build_environment(self):
        env = self.environment.build(self)
        print(env)
        return env

    def build_agent(self):
        agent = self.agent.build(self)
        print(agent)
        return agent

    def build_brain(self):
        brain = self.brain.build(self)
        print(brain)
        return brain

    def build_memory(self):
        memory = self.memory.build(self)
        print(memory)
        return memory

    def build_network(self):
        nnet = self.nnet.build(self)
        print(nnet)
        import os
        if self.args.get('load_weight', False) \
                and os.path.isfile(self.test.weight_file_name):
            nnet.load_weights(self.test.weight_file_name)
            self.args['base_epsilon'] = 0.001
        return nnet


if __name__ == '__main__':
    from tengu.drlfx.base_rl.sample.test_gym import TestCartPole

    test = TestCartPole()
    test.save_weights = False
    test.reset()

    env = NNetBuilder(test, "DDQN", nnet=DuelingNNet).build_environment()
    env.run()
