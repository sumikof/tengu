def oanda_rest_api(account,instrument,action,count,granularity):

    from oandapyV20 import API
    import oandapyV20.endpoints.instruments as instruments

    params = {
      "count": count,
      "granularity": granularity
    }

    api = API(access_token=account.token)

    # APIから為替レートのストリーミングを取得
    r = instruments.InstrumentsCandles(instrument=instrument,params=params)
    api.request(r)

    # 為替レートのdictをDataFrameへ変換
    return [ action(rate) for rate in r.response['candles'] ]
