from keras import Model, Input
import keras.backend as k
from keras.layers import Dense, Concatenate, Lambda
from keras.optimizers import Adam
import numpy as np

from tengu.drlfx.base_rl.base_abc import NNetABC
from tengu.drlfx.base_rl.loss_function import huberloss


class DuelingNNet(NNetABC):
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.input_size = state_size
        self.output_size = action_size

        main_input = Input(shape=(self.input_size,), name='main_input')
        hdn = Dense(hidden_size, activation='relu')(main_input)

        v = Dense(hidden_size, activation='relu')(hdn)
        v = Dense(1)(v)

        adv = Dense(hidden_size, activation='relu')(hdn)
        adv = Dense(self.output_size)(adv)

        model = Concatenate()([v, adv])
        model = Lambda(lambda a: k.expand_dims(a[:, 0], -1) + a[:, 1:] - k.mean(a[:, 1:], axis=1, keepdims=True),
                       output_shape=(self.output_size,))(model)

        self._model = Model(inputs=main_input, outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def predict(self, x):
        x = np.reshape(x, [len(x), self.input_size])
        return self._model.predict(x)

    def train_on_batch(self, x, y):
        x = np.reshape(x, [len(x), self.input_size])
        y = np.reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch(x, y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()

    def save_weights(self, file_name):
        self._model.save_weights(file_name)

    def load_weights(self, file_name):
        self._model.load_weights(file_name)


if __name__ == '__main__':
    from tengu.drlfx.base_rl.sample.test_gym import TestCartPole

    test = TestCartPole()
    from tengu.drlfx.base_rl.environment import EnvironmentDDQN

    ETA = 0.0001  # 学習係数
    hidden = 32

    from tengu.drlfx.base_rl.agent import AgentDDQN
    from tengu.drlfx.base_rl.brain import BrainDDQN

    Net = DuelingNNet
    brain = BrainDDQN(test,
                      main_network=Net(learning_rate=ETA,
                                       state_size=test.num_status,
                                       action_size=test.num_actions,
                                       hidden_size=hidden),
                      target_network=Net(learning_rate=ETA,
                                         state_size=test.num_status,
                                         action_size=test.num_actions,
                                         hidden_size=hidden)
                      )
    agent = AgentDDQN(brain)
    env = EnvironmentDDQN(test, agent)
    env.run()
