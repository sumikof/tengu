from keras import Model, Input
from keras.layers import Dense, Concatenate, Lambda, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import keras.backend as k
import numpy as np

from tengu.drlfx.base_rl.base_abc import NNetABC
from tengu.drlfx.base_rl.loss_function import huberloss
from tengu.drlfx.test_oanda import ACTION_SIZE

from logging import getLogger

logger = getLogger(__name__)


def edim(a):
    return k.expand_dims(a[:, 0], -1) + a[:, 1:] - k.mean(a[:, 1:], axis=1, keepdims=True)


class OandaNNet(NNetABC):
    def __init__(self, learning_rate=0.01, rate_size=32, position_size=3, model=None):
        super().__init__()
        self.output_size = ACTION_SIZE
        self.input_rate_size = rate_size
        self.input_position_size = position_size

        self._model = model or self.make_model()

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self._model.compile(loss=huberloss, optimizer=self.optimizer)

    def input_data_format(self, lst):
        rates = []
        positions = []
        for s in lst:
            rate = np.reshape(s.rates.map, (self.input_rate_size, self.input_rate_size, 1))
            rates.append(rate)

            positions.append(s.position)
        return rates, positions

    def predict(self, x):
        rates, positions = self.input_data_format(x)
        return self._model.predict([rates, positions])

    def train_on_batch(self, x, y):
        rates, positions = self.input_data_format(x)
        y = np.reshape(y, [len(y), self.output_size])
        return self._model.train_on_batch([rates, positions], y)

    def set_weights(self, w):
        return self._model.set_weights(w)

    def get_weights(self):
        return self._model.get_weights()

    def save_weights(self, file_name):
        self._model.save_weights(file_name)

    def load_weights(self, file_name):
        self._model.load_weights(file_name)

    def make_model(self):
        logger.debug("set default model")
        rates_input = Input(shape=(self.input_rate_size, self.input_rate_size, 1), name='rates_input')
        rate = Conv2D(64, kernel_size=(3, 3))(rates_input)
        rate = Activation('relu')(rate)
        rate = Conv2D(64, kernel_size=(3, 3))(rate)
        rate = Activation('relu')(rate)
        rate = MaxPooling2D(pool_size=(2, 2))(rate)
        rate = Dropout(0.25)(rate)
        rate = Dense(64, activation='relu')(rate)
        rate = Flatten()(rate)

        position_input = Input(shape=(self.input_position_size,), name='position_input')
        position = Dense(16)(position_input)

        main_input = Concatenate()([rate, position])
        main_input = Dense(128)(main_input)

        hdn = Dense(32, activation='relu')(main_input)

        v = Dense(32, activation='relu')(hdn)
        v = Dense(1)(v)

        adv = Dense(32, activation='relu')(hdn)
        adv = Dense(self.output_size)(adv)

        model = Concatenate()([v, adv])
        model = Lambda(edim, output_shape=(self.output_size,))(model)

        return Model(inputs=[rates_input, position_input], outputs=model)


if __name__ == '__main__':
    import datetime

    nnet = OandaNNet()
    yamlstr = nnet.model.to_yaml()
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ofile_name = "oanda_nnet_{}.yaml".format(now)
    #    open(ofile_name,'w').write(yamlstr)
    from keras.engine.saving import model_from_yaml

    load_model = model_from_yaml('oanda_nnet_20200609_202154.yaml')
    load_nnet = OandaNNet(model=load_model)
    load_nnet.model.summary()
