from keras import Model, Input
from keras.layers import Dense
from keras.optimizers import Adam

from tengu.drlfx.base_rl.base_abc import NNetABC
from tengu.drlfx.base_rl.modules.loss_function import huberloss
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F



class SimpleTorchNNet(nn.Module):
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        super(SimpleTorchNNet, self).__init__()
        self.input_size = state_size
        self.output_size = action_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, self.output_size)

        self.model.add_module('fc1', nn.Linear(hidden_size, self.output_size))
        print(self.model)

        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, input):
        h = self.ac1(self.fc1(input))
        h = self.ac2(self.fc2(h))
        out = self.fc3(h)
        return out

    def train_on_batch(self, x, y):
        self.model.train()
        loss = F.smooth_l1_loss()

    def set_weights(self, w):
        pass

    def get_weights(self):
        pass

    def save_weights(self, file_name):
        pass

    def load_weights(self, file_name):
        pass

    def predict(self, x):
        self.model.eval()


class SimpleNNet(NNetABC):
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        super().__init__()
        self.input_size = state_size
        self.output_size = action_size

        main_input = Input(shape=(self.input_size,), name='main_input')
        model = Dense(hidden_size, activation='relu')(main_input)
        model = Dense(hidden_size, activation='relu')(model)
        model = Dense(self.output_size, activation='linear')(model)

        self._model = Model(inputs=main_input, outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    @classmethod
    def build(cls, builder):
        return SimpleNNet(
            learning_rate=builder.args.get('learning_rate', 0.01),
            state_size=builder.test.num_status,
            action_size=builder.test.num_actions,
            hidden_size=builder.args.get('hidden_size', 10))

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
    from logging import basicConfig, INFO

    basicConfig(level=INFO)
    from tengu.drlfx.base_rl.sample.test_gym import TestCartPole
    test = TestCartPole()
    test.save_weights = False
    test.reset()

    from tengu.drlfx.base_rl.nnet_builder.nnet_builder import NNetBuilder
    env = NNetBuilder(test, "DDQN", nnet=SimpleNNet).build_environment()
    env.run()
