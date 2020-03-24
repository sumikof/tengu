import pandas as pd

def rateDF(access_token, instrument,params):

    from oandapyV20 import API
    import oandapyV20.endpoints.instruments as instruments
    api = API(access_token=access_token)

    # APIから為替レートのストリーミングを取得
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
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

    return rate


def main():
    import configparser
    config = configparser.ConfigParser()
    config.read('oanda.conf')

    accountID = config['OANDA']['accountID']
    access_token = config['OANDA']['access_token']

    instrument ="USD_JPY"
    # 200件5分足
    params = {
      "count": 3,
      "granularity": "M1"
    }

    rateDF(access_token=access_token ,instrument=instrument ,params=params)


if __name__ == '__main__':
    main()