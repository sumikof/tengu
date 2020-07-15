from tengu.drlfx.base_rl.experience_memory.ReplayMemory import ReplayMemory
import numpy as np
from logging import getLogger
from collections import namedtuple

Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))

logger = getLogger(__name__)


class BrainDDQN:
    def __init__(self, task, memory, main_network, target_network, batch_size, gamma, base_epsilon):

        self.task = task

        self.batch_size = batch_size
        self.memory = memory or ReplayMemory(10000)

        self.gamma = gamma  # 時間割引率
        self.base_epsilon = base_epsilon

        self.main_q_network = main_network
        self.target_q_network = target_network
        logger.info("brain memory type = {}".format(self.memory))

    @classmethod
    def build(cls, builder):

        main_network = builder.build_network()
        target_network = builder.build_network()
        memory = builder.build_memory()

        return BrainDDQN(builder.test,
                         memory,
                         main_network,
                         target_network,
                         builder.args.get('batch_size', 32),
                         builder.args.get('gamma', 0.99),
                         builder.args.get('base_epsilon', 0.99))

    def decide_action(self, state, episode, mask):
        """
        ε-greedy法で徐々に最適行動のみを採用する
        """
        epsilon = self.base_epsilon * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            target_action = self.main_q_network.predict([state])[0]
            target_action = target_action * mask
            action = np.max(np.argwhere(target_action == np.max(target_action[np.nonzero(target_action)])))
        else:
            action = np.random.choice(np.arange(self.task.num_actions)[mask])  # どれかのアクションを返す
        return action

    def replay(self):
        """
        memoryに保存したデータを使用しmainネットワークを更新する
        :return:
        """

        # データが溜まっていない間は実行しない
        if len(self.memory) < self.batch_size:
            return

        # ミニバッチの作成
        # ミニバッチの作成 メモリからミニバッチ分のデータを取り出す
        indexs, transitions = self.memory.sample(self.batch_size)
        #        batch = Transition(*zip(*transitions))

        # 教師信号Q(s_t,a_t)を求める
        (states, action_values) = self.get_expected_state_action_values(indexs, transitions)

        # 結合パラメータの更新
        self.update_main_q_network(states, action_values)

    def memorize(self, state, action, next_state, reward):
        self.memory.add(Transition(state, action, next_state, reward))

    def get_expected_state_action_values(self, indexs, batch):
        """
        sample batchからmain ネットワークの更新に使用するデータを作成する
        :param indexs:
        :param batch: memory
        :return: 状態,状態に対して更新するaction_value
        """

        states = []
        action_values = []
        for i, transition in enumerate(batch):
            if not self.task.check_status_is_done(transition.next_state):

                # 価値の計算
                main_q = self.main_q_network.predict([transition.next_state])[0]
                next_action = np.argmax(main_q)

                next_action_q = self.target_q_network.predict([transition.next_state])
                reward = transition.reward + self.gamma * next_action_q[0][next_action]

            else:
                reward = transition.reward

            states.append(transition.state)
            action_values.append(self.main_q_network.predict([transition.state])[0])

            reward_diff = action_values[i][transition.action] - reward

            if indexs is None:
                self.memory.update(None, transition, reward_diff)
            else:
                self.memory.update(indexs[i], transition, reward_diff)

            action_values[i][transition.action] = reward

        return states, action_values

    def update_main_q_network(self, states, action_values):
        # Qネットワークの重みを学習・更新する replay
        if len(self.memory) > self.batch_size:
            self.main_q_network.train_on_batch(
                states, action_values)

    def update_target_q_network(self):
        # target ネットワークを更新する
        self.target_q_network.set_weights(self.main_q_network.get_weights())

    def save_weights(self, file_name):
        self.main_q_network.save_weights(file_name)
