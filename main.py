import os

from tengu.oanda_api import oanda_rest_api


def printdump(rate):
    import json
    print(json.dumps(rate, indent=2))


def rest_action(account):
    action = printdump
    oanda_rest_api(account, instrument="USD_JPY", action=action, count=100, granularity="M1")


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
        outlier_min = q1 - (iqr * 1.5)
        outlier_max = q3 + (iqr * 1.5)

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


def main():
    from tengu.oanda_action.oanda_dataframe import oanda_dataframe
    df_org = oanda_dataframe('USD_JPY_M1.csv')
    from tengu.drlfx.base_rl.oanda_rl.oanda_agent import run_agent57,env_parameter
    env_parameter.rate_list = df_org['close'].values.tolist()
    run_agent57(enable_train=True)



if __name__ == '__main__':
    main()
