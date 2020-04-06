from scipy.stats import stats

import dl
from dl import util
from dl.oanda_keras import oanda_keras
from oanda_api import oanda_rest_api


class oanda_rest_json:
    def __init__(self):
        self.a = []

    def printdump(self,rate):
        import json
        print(json.dumps(rate, indent=2))

def rest_action(account):
    action = oanda_rest_json()
    oanda_rest_api(account, instrument="USD_JPY", action=action.printdump, count=100, granularity="M1")

def create_ex_csv(account):
    from oanda_action.create_csv import create_csv
    create_csv(account,instrument="USD_JPY",from_date="2020-01-01T00:00:00")

def get_account():
    from oanda_api.account import create_account
    account = create_account('oanda.conf')
    return account

def oanda_dataframe():
    import pandas as pd
    df = pd.read_csv('USD_JPY_M1.csv')
    df['date'] = pd.to_datetime(df['date'])
    #df = df[df['date'].dt.minute % 10 == 0]

    return df


def oanda_graph(df):
    import matplotlib.pyplot as plt

    plt.plot(df['date'], df['high'],color = 'red')
    plt.plot(df['date'], df['low'], color='blue')
    plt.show()

def outlier_iqr(df):

    for i in range(len(df.columns)):

        # 列を抽出する
        col = df.iloc[:,i]

        # 四分位数
        q1 = col.describe()['25%']
        q3 = col.describe()['75%']
        iqr = q3 - q1 #四分位範囲

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
    df = aft/org
    df = outlier_iqr(df).dropna()
    return df

import matplotlib.pyplot as plt
def main():
    df = oanda_dataframe()
#    from dl import sample_rnn
#    sample_rnn.odlprint()
    from dl import oanda_keras as odl
    #df = df[df['date'].dt.minute % 5 == 0]
    #df = df.tail(4000)
    #print(df.tail(5))
    df = df.drop('date', axis=1)[["close"]]


    df = util.standard_0_1(df)
    df = df.rolling(window=5).mean().dropna()
    #fig(df)
    #df = df.apply(stats.zscore, axis=0)
    after_proc(df)

def fig(df):
    df.plot(figsize=(15, 5))
    plt.show()

def after_proc(df):

    ok = oanda_keras(
        length_of_sequences = 5,
        in_out_neurons = 1,
        hidden_neurons = 300,
        batch_size=10000,
        epochs=5,
        validation_split=0.05)
    X_test, y_test, predicted,history = ok.oadna_make_batchdata(df)
    import pandas as pd
    dataf = pd.DataFrame(predicted)
    dataf.columns = ["predict"]
    dataf["input"] = y_test

    a = pd.DataFrame()
    a['val_loss'] = pd.Series(history.history['val_loss'])
    a['loss'] = pd.Series(history.history['loss'])
    fig(a)


if __name__ == '__main__':
    main()
