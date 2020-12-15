import gym
from keras.optimizers import Adam

import traceback
import os

from .agent57 import Agent57
from .model import InputType, DQNImageModel, LstmType
from .policy import AnnealingEpsilonGreedy
from .memory import PERRankBaseMemory, PERProportionalMemory

from .callbacks import ConvLayerView, MovieLogger
from .callbacks import LoggerType, TimeStop, TrainLogger, ModelIntervalCheckpoint
from .callbacks import DisTrainLogger, DisSaveManager


def run_gym_agent57(
        enable_train,
        env,
        env_name,
        kwargs,
        nb_trains=999_999_999,
        nb_time=999_999_999,
        logger_type=LoggerType.TIME,
        log_interval=0,
        test_env=None,
        test_episodes=10,
        is_load_weights=False,
        checkpoint_interval=0,
        movie_save=False,
    ):
    base_dir = os.path.join("tmp_{}".format(env_name))
    os.makedirs(base_dir, exist_ok=True)
    print("nb_time  : {:.2f}m".format(nb_time/60))
    print("nb_trains: {}".format(nb_trains))
    weight_file = os.path.join(base_dir, "{}_weight.h5".format(env_name))

    manager = Agent57(**kwargs)

    if test_env is None:
        test_actor = None
    else:
        test_actor = kwargs["actors"][0]
    log = DisTrainLogger(
        logger_type,
        interval=log_interval,
        savedir=base_dir,
        test_actor=test_actor,
        test_env=test_env,
        test_episodes=test_episodes,
        test_save_max_reward_file=os.path.join(base_dir, 'max_{step:02d}_{reward}.h5')
    )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        save_manager = DisSaveManager(
            save_dirpath=base_dir,
            is_load=is_load_weights,
            save_memory=False,
            checkpoint=(checkpoint_interval>0),
            checkpoint_interval=checkpoint_interval,
            verbose=0
        )

        manager.train(nb_trains, nb_time, callbacks=[save_manager, log])

    # 訓練結果を見る
    agent = manager.createTestAgent(kwargs["actors"][0], "tmp_{}/last/learner.dat".format(env_name))
    if agent is None:
        return
    agent.test(env, nb_episodes=5, visualize=False)

    env.close()



