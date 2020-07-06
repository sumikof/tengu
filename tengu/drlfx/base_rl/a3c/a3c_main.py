from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam
from tengu.drlfx.base_rl.loss_function import huberloss
import numpy as np
from logging import getLogger

from tengu.drlfx.base_rl.sample.test_gym import TestCartPole

logger = getLogger(__name__)


class A3CSimpeNNet:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.input_size = state_size
        self.output_size = action_size

        main_input = Input(batch_shape=(None, self.input_size), name='main_input')
        l_dense = Dense(hidden_size, activation='relu')(main_input)
        out_actions = Dense(self.output_size, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        self._model = Model(inputs=main_input, outputs=[out_actions, out_value])

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam

    def compile(self):
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, w):
        self._model.set_weights(w)

    def predict(self, states):
        logger.debug("predict input {}".format(states))
        s = np.reshape(states, (len(states), 4))
        x = self._model.predict(s)
        return x

    def train_on_batch(self, feed_dict):
        self._model.train_on_batch(feed_dict['state'], [feed_dict['action'], feed_dict['reward']])

    def _make_predict_function(self):
        self._model._make_predict_function()


def build_model():
    model = A3CSimpeNNet()
    return model


class ParameterServer:
    def __init__(self, model_generator):
        self._model_generator = model_generator
        self._model = self.build_model()

    def pull_weights(self):
        return self._model.get_weights()

    def build_model(self):
        return self._model_generator()

    def update_weight(self, feed_dict):
        self._model.train_on_batch(feed_dict)


class GlobalStatus:
    def __init__(self):
        self.is_learned = False
        self.gamma_n = 0.1


class A3cBrain:
    def __init__(self, thread_name: str, param_server: ParameterServer, agent, memory, batch_size=32,
                 gamma=0.99,
                 base_epsilon=0.5):
        self.thread_name = thread_name
        self.param_server = param_server
        self.model = self.param_server.build_model()
        self.model._make_predict_function()
        self.model.compile()
        self.agent = agent

        self.batch_size = batch_size
        self.memory = memory

        self.gamma = gamma  # 時間割引率
        self.base_epsilon = base_epsilon

        self.train_queue = [[], [], [], [], []]

    @property
    def task(self):
        return self.agent.task

    def pull_parameter_server(self):
        weight = self.param_server.pull_weights()
        self.model.set_weights(weight)

    def update_parameter_server(self):
        if len(self.train_queue[0]) < self.batch_size:
            return

        state, action, reward, next_state, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        state = np.vstack(state)
        action = np.vstack(action)
        reward = np.vstack(reward)
        print(next_state)
        next_state = np.vstack(next_state)
        print(next_state)
        s_mask = np.vstack(s_mask)

        # Nステップあとの状態s_から、その先得られるであろう時間割引総報酬vを求めます
        _, v = self.model.predict(next_state)

        reward = reward + self.gamma * v * s_mask
        feed_dict = {'state': state, 'action': action, 'reward': reward}
        self.param_server.update_weight(feed_dict)

    def decide_action(self, state, episode):
        """
        ε-greedy法で徐々に最適行動のみを採用する
        """
        epsilon = self.base_epsilon * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            target_action, v = self.model.predict([state])
            target_action = target_action[0]
            target_action = target_action * task.mask
            action = np.max(np.argwhere(target_action == np.max(target_action[np.nonzero(target_action)])))
        else:
            action = np.random.choice(np.arange(self.task.num_actions)[task.mask])  # どれかのアクションを返す
        return action

    def train_push(self, state, action, reward, next_state):
        self.train_queue[0].append(state)
        self.train_queue[1].append(action)
        self.train_queue[2].append(reward)

        if next_state is None:
            self.train_queue[3].append(self.task.blank_status)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(next_state)
            self.train_queue[4].append(1.)


class A3cAgent:
    def __init__(self, thread_name, param_server, env, memory, batch_size=32, gamma=0.99,
                 base_epsilon=0.5):
        self.env = env
        self.brain = A3cBrain(thread_name, param_server, self, memory, batch_size, gamma,
                              base_epsilon)
        self.memory = []
        self.r = 0  # 時間割引した、「いまからNステップ分あとまで」の総報酬R
        self.num_step_return = 3

    @property
    def task(self):
        return self.env.task

    def get_action(self, state, episode):
        """行動の決定"""
        action = self.brain.decide_action(state, episode)
        return action

    def advantage_push_local_brain(self, state, action, reward, next_state):
        def get_sample(memory, n):
            state, action, _, _ = memory[0]
            _, _, _, next_state = memory[n - 1]
            return state, action, self.r, next_state

        # one-hotコーディングにしたa_catsをつくり、、s,a_cats,r,s_を自分のメモリに追加
        a_cats = np.zeros(self.task.num_actions)
        a_cats[action] = 1
        self.memory.append((state, a_cats, reward, next_state))

        self.r = (self.r + reward * self.brain.gamma) / self.brain.gamma

        if next_state is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                state, action, reward, next_state = get_sample(self.memory, n)
                self.brain.train_push(state, action, reward, next_state)
                self.r = (self.r - self.memory[0][2] / self.brain.gamma)
                self.memory.pop(0)

            self.r = 0

        if len(self.memory) >= self.num_step_return:
            state, action, reward, next_state = get_sample(self.memory, self.num_step_return)
            self.brain.train_push(state, action, reward, next_state)
            self.r = self.r - self.memory[0][2]
            self.memory.pop(0)


class A3cEnvironment:
    def __init__(self, thread_name, param_server, global_status, task, max_steps=200, train_interval=20):
        self.global_status = global_status
        self.agent = A3cAgent(thread_name, param_server, self, memory=None)
        self.task = task

        self.max_steps = max_steps
        self.step = 0
        self.episode = 0
        self.train_interval = train_interval

    def run(self):
        self.agent.brain.pull_parameter_server()
        state = self.task.reset()
        done = False
        while not done:
            logger.debug("step = {} start".format(self.step))
            if self.max_steps != 0 and self.step > self.max_steps:
                break

            # main networkから行動を決定
            action = self.agent.get_action(state, self.episode)
            logger.debug("get action {}".format(action))

            # 行動の実行（次の状態と終了判定を取得）
            next_state, reward, done, info = self.task.step(action, self)
            logger.debug("task.step done={}".format(done))

            # Advantageを考慮した報酬と経験を、localBrainにプッシュ
            # メモリに追加
            # self.agent.memorize(state, action, next_state, reward)
            self.agent.advantage_push_local_brain(state, action, reward, next_state)

            state = next_state

            # 終了 or 一定間隔で重みを更新
            if done or (self.step % self.train_interval == 0):
                self.agent.brain.update_parameter_server()
                self.agent.brain.pull_parameter_server()

            if done:
                # 終了条件を設定
                break

            self.step += 1

        if self.task.is_finish():
            self.global_status.is_learned = True


class WorkerThread:
    def __init__(self, thread_name, param_server, global_status, task):
        self.environment = A3cEnvironment(thread_name, param_server, global_status, task)
        self.global_status = global_status

    def run(self):
        while not self.global_status.is_learned:
            self.environment.run()


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG)

    frames = 0
    global_status = GlobalStatus()

    parm_serv = ParameterServer(build_model)
    threads = []

    local_thread_name = "local_thread"
    task = TestCartPole()
    threads.append(WorkerThread(local_thread_name, parm_serv, global_status, task))
    threads.append(WorkerThread(local_thread_name, parm_serv, global_status, task))

    for worker in threads:
        worker.run()
