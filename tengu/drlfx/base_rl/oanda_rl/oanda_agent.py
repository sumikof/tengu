from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from tengu.drlfx.base_rl.oanda_rl.oanda_environment import OandaEnv
from tengu.drlfx.base_rl.oanda_rl.oanda_processor import OandaProcessor

env = OandaEnv(rate_list=[i for i in range(1000)],test_size=100)
nb_actions = env.action_space.n

processor = OandaProcessor()

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dense(nb_actions, activation="linear"))

# Duel-DQNアルゴリズム関連の幾つかの設定
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# Duel-DQNのAgentクラスオブジェクトの準備 （上記processorやmodelを元に）
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type="avg", target_model_update=1e-2, policy=policy,
               processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])
print(dqn.model.summary())

# 定義課題環境に対して、アルゴリズムの学習を実行 （必要に応じて適切なCallbackも定義、設定可能）
# 上記Processorクラスの適切な設定によって、Agent-環境間の入出力を通して設計課題に対しての学習が進行
dqn.fit(env, nb_steps=100000, visualize=False, verbose=2)

# 学習後のモデルの重みの出力
#dqn.save_weights("duel_dqn_{}_weights.h5f".format("Pendulum-v0"), overwrite=True)

# 学習済モデルに対して、テストを実行 （必要に応じて適切なCallbackも定義、設定可能）
dqn.test(env, nb_episodes=100, visualize=True)