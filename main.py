

def run_agent57(enable_train):

    from tengu.common.parameter import TenguParameter,argument_config
    param = TenguParameter()
    argument_config(param.general_param)

    from tengu.drlfx.base_rl.oanda_rl.oanda_generator import OandaEnvGenerator
    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import env_manager
    env_manager.set_generator(OandaEnvGenerator(param.general_param))
    env = env_manager.create_env()

    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))


    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import MyActor1
    param.agent_param["actors"] = [MyActor1 for _ in range(param.general_param["actor_num"])]  # [MyActor1, MyActor2]

    from tengu.drlfx.base_rl.agent.callbacks import LoggerType
    from tengu.drlfx.base_rl.agent.main_runner import run_gym_agent57
    run_gym_agent57(
        enable_train,
        env,
        env_name="OandaEnv-v0",
        kwargs=param.agent_param,
        nb_trains=param.general_param["nb_trains"],  # 最大試行回数
        nb_time=param.general_param["nb_time"],  # 最大実行時間
        logger_type=LoggerType.STEP,
        log_interval=param.general_param["log_interval"],
        test_env=env_manager.create_env,
        is_load_weights=param.general_param["is_load_weights"]
    )
    env.close()


# ----------------------


if __name__ == '__main__':
    from logging import basicConfig,DEBUG

    basicConfig(level=DEBUG)
    # 複数Actorレーニング
    run_agent57(enable_train=True)
