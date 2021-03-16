from logging import getLogger

logger = getLogger(__name__)


class OandaEnvGenerator:

    def __init__(self, param):
        self.oanda_param = {
            "test_size": 60 * 24 * 5,
            "spread": 0.018,
            "err_handle_f": True
        }

        intersection_keys = self.oanda_param.keys() & param.keys()
        for key in intersection_keys:
            self.oanda_param[key] = param[key]

        from tengu.oanda_action.oanda_dataframe import oanda_dataframe
        df_org = oanda_dataframe(param["rate_csv"])

        self.rate_list = df_org['close'].values.tolist()

    def create_env(self, env_name=""):
        from tengu.drlfx.oanda_rl.oanda_environment import OandaEnv
        return OandaEnv(env_name, rate_list=self.rate_list, **self.oanda_param)


if __name__ == '__main__':
    import random

    act_pattern = [0, 1, 2, 3]
    param = {
        "test_size": 60 * 24 * 5,
        "spread": 0.018,
        "rate_csv": "../../../USD_JPY_M1.csv"
    }
    from logging import WARNING,DEBUG,basicConfig
    logarg = {
        "level": DEBUG,
        "format": "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
    }
    basicConfig(**logarg)

    gen = OandaEnvGenerator(param)
    env = gen.create_env("oanda_test")

    done = False
    logger.error("start")
    while (not done):
        action = random.choice(act_pattern)
        observe, reward, done, info = env.step(action)
        logger.error(reward)
    logger.error("finish")
