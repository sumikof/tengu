import pandas as pd
import matplotlib.pyplot as plt
import datetime

def main():
    import configparser
    config = configparser.ConfigParser()
    config.read('oanda.conf')

    accountID = config['OANDA']['accountID']
    access_token = config['OANDA']['access_token']

    # 200件5分足
    params = {
      "count": 200,
      "granularity": "M1"
    }

    access(access_token,params)


def access(access_token, params):

    from oandapyV20 import API
    import oandapyV20.endpoints.instruments as instruments
    api = API(access_token=access_token)

    # APIから為替レートのストリーミングを取得
    r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
    api.request(r)

    import json
    print(json.dumps(r.response, indent=2))
    # 為替レートのdictをDataFrameへ変換
    rate = pd.DataFrame.from_dict({r.response['candles'][i]['time']: r.response['candles'][i]['mid']
                               for i in range(0,len(r.response['candles']))
                               for j in r.response['candles'][i]['mid'].keys()},
                           orient='index',
                          )

    # APIから取得した日付（time）を日付型に変換
    rate.index = pd.to_datetime(rate.index)

    # 念のためDataFrameの確認
    print(rate.head())

if __name__ == '__main__':
    main()