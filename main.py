
def run_agent57(enable_train):
    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import EnvironmentGenerator

    class OandaEnvGenerator(EnvironmentGenerator):
        def __init__(self, rate_list):
            self.rate_list = rate_list

        def create_env(self):
            from tengu.drlfx.base_rl.oanda_rl.oanda_environment import OandaEnv

            return OandaEnv(rate_list=self.rate_list)

    from tengu.drlfx.base_rl.agent.common import seed_everything
    import time
    seed = int(time.time())
    seed_everything(seed)

    from tengu.drlfx.base_rl.agent.main_runner import run_gym_agent57

    from tengu.oanda_action.oanda_dataframe import oanda_dataframe
    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import env_manager
    df_org = oanda_dataframe('USD_JPY_M1.csv')
    env_generator = OandaEnvGenerator(df_org['close'].values.tolist())
    env_manager.set_generator(env_generator)
    env = env_manager.create_env()

    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    from tengu.common.parameter import TenguParameter,argument_config
    param = TenguParameter()
    argument_config(param.general_param)

    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import MyActor1
    param.agent_param["actors"] = [MyActor1 for _ in range(param.general_param["actor_num"])]  # [MyActor1, MyActor2]

    from tengu.drlfx.base_rl.agent.callbacks import LoggerType

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
    # 複数Actorレーニング
    run_agent57(enable_train=True)
