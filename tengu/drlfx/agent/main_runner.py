import os

from .agent57 import Agent57
from .callbacks import LoggerType
from .callbacks import DisTrainLogger, DisSaveManager, DisTrainDebugger


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
):
    from tengu.drlfx.agent.common import seed_everything
    import time
    seed_everything(int(time.time()))

    base_dir = os.path.join("tmp_{}".format(env_name))
    os.makedirs(base_dir, exist_ok=True)
    print("nb_time  : {:.2f}m".format(nb_time / 60))
    print("nb_trains: {}".format(nb_trains))
    weight_file = os.path.join(base_dir, "{}_weight.h5".format(env_name))

    kwargs["input_shape"] = env.observation_space.shape
    kwargs["nb_actions"] = env.action_space.n
    kwargs["demo_episode_dir"] = "tmp_{}.".format(env_name)

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
            checkpoint=(checkpoint_interval > 0),
            checkpoint_interval=checkpoint_interval,
            verbose=1
        )
        debugger = DisTrainDebugger()
        # callbacks = [save_manager, log, debugger]
        callbacks = [save_manager, debugger]
        manager.train(nb_trains, nb_time, callbacks=callbacks)

    # 訓練結果を見る
    agent = manager.createTestAgent(kwargs["actors"][0], "tmp_{}/last/learner.dat".format(env_name))
    if agent is None:
        return
    agent.test(env, nb_episodes=5, visualize=False)

    env.close()
