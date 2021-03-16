import numpy

from tengu.drlfx.oanda_rl.oanda_generator import OandaEnvGenerator
from logging import getLogger

from tengu.drlfx.oanda_rl.oanda_processor import OandaProcessor

logger = getLogger(__name__)

if __name__ == '__main__':
    numpy.random.seed(123)
    # ログ設定
    from logging import WARNING, DEBUG, basicConfig

    logarg = {
        "level": DEBUG,
        "format": "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
    }
    basicConfig(**logarg)

    # 環境構築
    env_param = {
        "test_size": 60 * 24 * 5,
        "spread": 0.018,
        "rate_csv": "../../../USD_JPY_M1.csv",
        "err_handle_f": False
    }
    gen = OandaEnvGenerator(env_param)
    env = gen.create_env("oanda_test")
    env.seed(123)

    nb_actions = 4

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten
    from keras.optimizers import Adam

    from rl.agents.dqn import DQNAgent
    from rl.policy import BoltzmannQPolicy
    from rl.memory import SequentialMemory

    # モデルの定義
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)  # experience reply で用いるmemory
    policy = BoltzmannQPolicy()  # 行動選択手法の定義

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)  # 学習。ここでnb_stepsは全エピソードのステップ数の合計が50000（だと思う）

    dqn.test(env, nb_episodes=5, visualize=True)
