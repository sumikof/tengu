import os


def argument_config(config):
    import argparse
    parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')  # 2. パーサを作る

    for key,val in config.items():
        parser.add_argument('--{}'.format(key),type=type(val))  # オプション引数（指定しなくても良い引数）を追加

    args = parser.parse_args()

    for key,val in vars(args).items():
        if val is not None:
            config[key] = val


def set_optimizer(kwargs):
    def convert_optimizer(arg, param):
        if arg[param]["optimizer"] == "Adam":
            from keras.optimizers import Adam
            arg[param] = Adam(**arg[param]["optimizer_argument"])
        else:
            raise NotImplementedError

    convert_optimizer(kwargs, "optimizer_ext")
    convert_optimizer(kwargs, "optimizer_int")
    convert_optimizer(kwargs, "optimizer_rnd")
    convert_optimizer(kwargs, "optimizer_emb")


def set_lstm_type(kwargs):
    from tengu.drlfx.base_rl.agent.model import LstmType
    if kwargs["lstm_type"] == "STATEFUL":
        kwargs["lstm_type"] = LstmType.STATEFUL
    else:
        raise NotImplementedError


def set_intrinsic_reward(kwargs):
    def get_intrinsic_reward(name):
        from tengu.drlfx.base_rl.agent.model import UvfaType
        if name == "ACTION":
            return UvfaType.ACTION
        elif name == "REWARD_EXT":
            return UvfaType.REWARD_EXT
        elif name == "REWARD_INT":
            return UvfaType.REWARD_INT
        elif name == "POLICY":
            return UvfaType.POLICY
        else:
            raise NotImplementedError

    def convert_uvfa(arg, key):
        arg[key] = [get_intrinsic_reward(i) for i in arg[key]]

    convert_uvfa(kwargs, "uvfa_ext")
    convert_uvfa(kwargs, "uvfa_int")


class TenguParameter:
    config_directory = 'config/'
    default_direcotry = 'default/'
    agent_parameter_file = 'agent_parameter.yaml'
    general_parametr_file = 'general_parameter.yaml'

    def __init__(self):
        self.agent_param = self.create_agent_parameter()
        self.general_param = self.create_general_parameter()

        # train
        self.agent_param["demo_ratio_steps"] = self.general_param["nb_trains"]
        # IS反映率の上昇step数
        self.agent_param["memory_kwargs"]["beta_steps"] = self.general_param["nb_trains"]

    def read_yaml(self, filenm):
        import yaml
        if os.path.isfile(self.config_directory + filenm):
            with open(self.config_directory + filenm) as file:
                kwargs = yaml.safe_load(file)
        else:
            kwargs = {}
        return kwargs

    def read_user_config(self, filenm):
        return self.read_yaml(filenm)

    def read_defalt_config(self, filenm):
        return self.read_yaml(self.default_direcotry + filenm)


    def read_parameter_file(self, filenm):
        config = self.read_defalt_config(filenm)
        user_config = self.read_user_config(filenm)

        self.update_config(config,user_config)

        return config

    def create_agent_parameter(self):
        kwargs = self.read_parameter_file(self.agent_parameter_file)
        set_optimizer(kwargs)
        set_lstm_type(kwargs)
        set_intrinsic_reward(kwargs)


        from tengu.drlfx.base_rl.agent.model import InputType
        from tengu.drlfx.base_rl.agent.model import ValueModel
        from tengu.drlfx.base_rl.oanda_rl.oanda_processor import OandaProcessor

        kwargs["input_type"] = InputType.VALUES
        kwargs["input_model"] = ValueModel(32, 1)

        # other
        kwargs["processor"] = OandaProcessor()

        return kwargs

    def create_general_parameter(self):
        return self.read_parameter_file(self.general_parametr_file)

    def update_config(self, config, user_config):
        for key,val in user_config.items():
            config[key] = val
