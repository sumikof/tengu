class OandaEnvGenerator:

    def __init__(self, param):
        self.oanda_param = {
            "state_size": 1,
            "test_size": 60 * 24 * 5,
            "spread": 0.018
        }

        intersection_keys = self.oanda_param.keys() & param.keys()
        for key in intersection_keys:
            self.oanda_param[key] = param[key]

        from tengu.oanda_action.oanda_dataframe import oanda_dataframe
        df_org = oanda_dataframe(param["rate_csv"])

        self.rate_list = df_org['close'].values.tolist()

    def create_env(self):
        from tengu.drlfx.base_rl.oanda_rl.oanda_environment import OandaEnv
        return OandaEnv(rate_list=self.rate_list,**self.oanda_param)

