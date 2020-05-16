from keras import Model, Input
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from dl.ddqn.loss_function import huberloss


class SimpleNNet:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.input_size = state_size
        self.output_size = action_size

        main_input = Input(shape=(self.input_size,), name='main_input')
        model = Dense(hidden_size, activation='relu')(main_input)
        model = Dense(hidden_size, activation='relu')(model)
        model = Dense(self.output_size, activation='linear')(model)

        self.model = Model(inputs=main_input, outputs=model)

        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

