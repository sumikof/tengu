import os

from tengu.oanda_api import oanda_rest_api


class oanda_rest_json:
    def __init__(self):
        self.a = []

    def printdump(self, rate):
        import json
        print(json.dumps(rate, indent=2))


def rest_action(account):
    action = oanda_rest_json()
    oanda_rest_api(account, instrument="USD_JPY", action=action.printdump, count=100, granularity="M1")


def create_ex_csv(account):
    from tengu.oanda_action.create_csv import create_csv
    create_csv(account, instrument="USD_JPY", from_date="2020-01-01T00:00:00")


def get_account():
    from tengu.oanda_api.account import create_account
    account = create_account('oanda.conf')
    return account


def oanda_graph(df):
    import matplotlib.pyplot as plt

    plt.plot(df['date'], df['high'], color='red')
    plt.plot(df['date'], df['low'], color='blue')
    plt.show()


def outlier_iqr(df):
    for i in range(len(df.columns)):
        # 列を抽出する
        col = df.iloc[:, i]

        # 四分位数
        q1 = col.describe()['25%']
        q3 = col.describe()['75%']
        iqr = q3 - q1  # 四分位範囲

        # 外れ値の基準点
        outlier_min = q1 - (iqr) * 1.5
        outlier_max = q3 + (iqr) * 1.5

        # 範囲から外れている値を除く
        col[col < outlier_min] = None
        col[col > outlier_max] = None

    return df


def henka(df):
    aft = df.drop(df.head(1).index).reset_index(drop=True)
    org = df.drop(df.tail(1).index).reset_index(drop=True)
    df = aft / org
    df = outlier_iqr(df).dropna()
    return df


import matplotlib.pyplot as plt


def main():
    from tengu.oanda_action.oanda_dataframe import oanda_dataframe
    from tengu.drlfx.test_oanda import TestOanda
    from tengu.drlfx.base_rl.agent import AgentDDQN
    from tengu.drlfx.base_rl.brain import BrainDDQN
    from tengu.drlfx.oanda_nnet import OandaNNet

    df_org = oanda_dataframe('USD_JPY_M1.csv')
    rate_size = 64

    test = TestOanda(df_org['close'].values, (60 * 24 * 5), rate_size)
    test.save_weights = True

    eta = 0.0001  # 学習係数

    main_network = OandaNNet(learning_rate=eta, rate_size=rate_size)
    target_network = OandaNNet(learning_rate=eta, rate_size=rate_size)
    base_epsilon = 0.5

    if os.path.isfile(test.weight_file_name):
        main_network.load_weights(test.weight_file_name)
        target_network.load_weights(test.weight_file_name)
        base_epsilon = 0.1

    brain = BrainDDQN(test,
                      main_network=main_network,
                      target_network=target_network,
                      base_epsilon=base_epsilon)
    agent = AgentDDQN(brain)

    from tengu.drlfx.base_rl.environment import EnvironmentDDQN
    env = EnvironmentDDQN(test, agent, num_episodes=500, max_steps=0)
    env.run()


def fig(df):
    df.plot(figsize=(15, 5))
    plt.show()


if __name__ == '__main__':
    main()
