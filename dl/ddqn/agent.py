from dl.ddqn.brain import BrainDDQN

class AgentDDQN:
    def __init__(self, brain):
        self.brain = brain

    def update_Q_function(self):
        """memoryからmain Q networkの更新"""
        self.brain.replay()

    def get_action(self, state, episode):
        """行動の決定"""
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)

    def update_target_Q_function(self):
        """target Q networkをmain Q networkと同じになるように更新"""
        self.brain.update_target_q_network()