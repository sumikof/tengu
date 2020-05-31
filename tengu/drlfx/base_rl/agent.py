class AgentDDQN:
    def __init__(self, brain):
        self.brain = brain

    def update_Q_function(self):
        """memoryからmain Q networkの更新"""
        self.brain.replay()

    def get_action(self, state, episode, mask):
        """行動の決定"""
        action = self.brain.decide_action(state, episode, mask)
        return action

    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)

    def update_target_Q_function(self):
        """target Q networkをmain Q networkと同じになるように更新"""
        self.brain.update_target_q_network()

    def save_weights(self,file_name):
        self.brain.save_weights(file_name)