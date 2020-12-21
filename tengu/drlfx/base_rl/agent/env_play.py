import glob
import os
import pickle

import numpy as np

from .actor import Actor
from .policy import EpsilonGreedy


def add_memory(episode_save_dir, memory, model_builder, kwargs):
    actor = Actor(
        kwargs["input_shape"],
        kwargs["input_sequence"],
        kwargs["nb_actions"],
        EpsilonGreedy(0),
        kwargs["batch_size"],
        kwargs["lstm_type"],
        kwargs["reward_multisteps"],
        kwargs["lstmful_input_length"],
        kwargs["burnin_length"],
        kwargs["enable_intrinsic_actval_model"],
        kwargs["enable_rescaling"],
        kwargs["priority_exponent"],
        kwargs["int_episode_reward_k"],
        kwargs["int_episode_reward_epsilon"],
        kwargs["int_episode_reward_c"],
        kwargs["int_episode_reward_max_similarity"],
        kwargs["int_episode_reward_cluster_distance"],
        kwargs["int_episodic_memory_capacity"],
        kwargs["rnd_err_capacity"],
        kwargs["rnd_max_reward"],
        kwargs["policy_num"],
        kwargs["test_policy"],
        kwargs["beta_max"],
        kwargs["gamma0"],
        kwargs["gamma1"],
        kwargs["gamma2"],
        kwargs["ucb_epsilon"],
        kwargs["ucb_beta"],
        kwargs["ucb_window_size"],
        model_builder,
        kwargs["uvfa_ext"],
        kwargs["uvfa_int"],
        0,
    )
    actor.build_model(None)
    step_interval = kwargs["step_interval"]
    input_shape = kwargs["input_shape"]

    for fn in glob.glob(os.path.join(episode_save_dir, "episode*.dat")):
        print("load: {}".format(fn))
        with open(fn, 'rb') as f:
            epi_states = pickle.load(f)
        if len(epi_states) <= 0:
            continue

        if input_shape != np.asarray(epi_states[0]["observation"]).shape:
            print("episode shape is not match. input_shape{} != epi_shape{}".format(
                input_shape,
                epi_states[0]["observation"].shape
            ))
            continue

        # init
        actor.training = True
        actor.episode_begin()

        # episode
        total_reward = 0
        for step, epi_state in enumerate(epi_states):

            if step % step_interval == 0:
                actor.forward_train_before(epi_state["observation"])
                exp = actor.create_exp(False)
                memory.add(exp)
                actor.forward_train_after()

                # アクションを入れかえる
                actor.recent_actions[-1] = epi_state["action"]

                actor.backward(epi_state["reward"], epi_state["done"])

            # 最後の状態も追加
            if epi_state["done"]:
                actor.forward_train_before(epi_state["observation"])
                exp = actor.create_exp(False)
                memory.add(exp)

            # 表示用
            total_reward += epi_state["reward"]

        print("demo replay loaded, on_memory: {}, total reward: {}".format(len(memory), total_reward))
