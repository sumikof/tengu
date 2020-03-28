
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
    df = df.tail(1000)

    return df


def oanda_graph(df):
    import matplotlib.pyplot as plt

    plt.plot(df['date'], df['high'],color = 'red')
    plt.plot(df['date'], df['low'], color='blue')
    plt.show()


def main():
    df = oanda_dataframe()
    print(df)
    oanda_graph(df)




if __name__ == '__main__':
    main()
