from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from dl.ddqn.replay_memory import Transition, ReplayMemory
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
CAPACITY = 10000

NUM_DIZITIZED = 1096  # 状態の数
ETA = 0.0001  # 学習係数
GAMMA = 0.99  # 時間割引率


# [1]損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    from keras import backend as K
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class NNet:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)


class BrainDDQN:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 取れる行動の数
        self.num_states = num_states
        self.memory = ReplayMemory(CAPACITY)

        learning_rate = ETA
        state_size = self.num_states
        action_size = self.num_actions
        hidden_size = 10

        self.main_q_network = NNet(learning_rate, state_size, action_size, hidden_size)
        self.target_q_network = NNet(learning_rate, state_size, action_size, hidden_size)

    def decide_action(self, state, episode):
        """
        ε-greedy法で徐々に最適行動のみを採用する
        """
        epsilon = 0.001 * (0.9 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            target_action = self.main_q_network.model.predict(state)[0]
            action = np.argmax(target_action)
        else:
            action = np.random.choice(self.num_actions)  # どれかのアクションを返す

        return action

    def replay(self):
        """
        memoryに保存したデータを使用しmainネットワークを更新する
        :return:
        """

        # データが溜まっていない間は実行しない
        if len(self.memory) < BATCH_SIZE:
            return

        # ミニバッチの作成
        # ミニバッチの作成 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)
#        batch = Transition(*zip(*transitions))

        # 教師信号Q(s_t,a_t)を求める
        (states, action_values) = self.get_expected_state_action_values(transitions)

        # 結合パラメータの更新
        self.update_main_q_network(states, action_values)

    def get_expected_state_action_values(self, batch):
        """
        sample batchからmain ネットワークの更新に使用するデータを作成する
        :param batch: memory
        :return: 状態,状態に対して更新するaction_value
        """

        states = np.zeros((BATCH_SIZE, self.num_states))
        action_values = np.zeros((BATCH_SIZE, self.num_actions))

        for i, (state_b, action_b,next_state_b, reward_b ) in enumerate(batch):

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値の計算
                main_q = self.main_q_network.model.predict(next_state_b)
                next_action = np.argmax(main_q)

                next_action_q = self.target_q_network.model.predict(next_state_b)
                reward = reward_b + GAMMA * next_action_q[0][next_action]

            else:
                reward = reward_b

            states[i] = state_b

            action_values[i] = self.main_q_network.model.predict(state_b)
            action_values[i][action_b] = reward

        return states, action_values

    def update_main_q_network(self, states, action_values):
        # Qネットワークの重みを学習・更新する replay
        if len(self.memory) > BATCH_SIZE:
            self.main_q_network.model.train_on_batch(
                states, action_values)

    def update_target_q_network(self):
        # target ネットワークを更新する
        self.target_q_network.model.set_weights(self.main_q_network.model.get_weights())
