def setup_loglevel(param):
    from logging import basicConfig, getLogger
    logarg = {
        "level":param.general_param["basic_loglevel"],
        "format": "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
    }
    basicConfig(**logarg)
    for module in param.general_param["module_loglevel"]:
        getLogger(module["module_name"]).setLevel(level=module["loglevel"])

def logformat():
    pass

def run_agent57(enable_train):
    from tengu.common.parameter import TenguParameter, argument_config
    param = TenguParameter()
    argument_config(param.general_param)

    from tengu.drlfx.oanda_rl.oanda_generator import OandaEnvGenerator
    env_generator = OandaEnvGenerator(param.general_param)

    env = env_generator.create_env("main")
    setup_loglevel(param)
    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("observation_space.shae : " + str(env.observation_space.shape))
    print("reward_range      : " + str(env.reward_range))

    from tengu.drlfx.oanda_rl.oanda_agent import MyActor
    param.agent_param["actors"] = [MyActor for _ in range(param.general_param["actor_num"])]  # [MyActor1, MyActor2]
    param.agent_param["actor_args"] = env_generator

    from tengu.drlfx.agent.callbacks import LoggerType
    from tengu.drlfx.agent.main_runner import run_gym_agent57
    run_gym_agent57(
        enable_train,
        env,
        env_name="OandaEnv-v0",
        kwargs=param.agent_param,
        nb_trains=param.general_param["nb_trains"],  # 最大試行回数
        nb_time=param.general_param["nb_time"],  # 最大実行時間
        logger_type=LoggerType.STEP,
        log_interval=param.general_param["log_interval"],
        test_env=env_generator.create_env,
        is_load_weights=param.general_param["is_load_weights"]
    )
    env.close()


# ----------------------


if __name__ == '__main__':
    # 複数Actorレーニング
    run_agent57(enable_train=True)
