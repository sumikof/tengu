import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
from tengu.oanda_api import strformatdate


def create_csv(account,instrument,from_date):

    oanda = oandapyV20.API(access_token=account.token)

    params1 = {"instruments": instrument}
    psnow = pricing.PricingInfo(accountID=account.id, params=params1)
    now = oanda.request(psnow)  # 現在の価格を取得

    end = now['time']

    print('start date => ', end)

    asi = 'M1'  # 取得した時間足を指定
    get_date = from_date  # どの期間まで取得したいか指定(年-月-日T時:分の形式で指定、時間はなくても良い(おそらく))
    # "2005-01-02T19:12:00" からが良い (アルゴリズム取引をする場合は、2010年ごろからがオススメ)

    i = 0
    while (end > get_date):
        params = {"count": 5000, "granularity": asi, "to": end}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params, )
        apires = oanda.request(r)
        res = r.response['candles']
        end = res[0]['time']
        n = 0
        if i == 0:
            res1 = res
        else:
            for j in range(n, len(res1)): res.append(res1[j])
            if end < get_date:
                for j in range(5000):
                    if res[j]['time'] > get_date:
                        end = res[j - 1]['time']
                        n = j
                        break
        res1 = res[n:]
        if i % 100 == 0:
            print('res ok', i + 1, 'and', 'time =', res1[0]['time'])
        i += 1

    print('GET Finish!', i * 5000 - n)  # どのくらいデータを取得したか確認

    data = []
    # 少し形を成形してあげる
    for raw in res1:
        data.append([raw['time'][:-4], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])

    # DataFrameに変換して、CSVファイルに保存をする。
    df = pd.DataFrame(data)
    df.columns = ['date', 'open', 'high', 'low', 'close']

    # 時間を全て日本時間に変更する。
    for i in df['date']:
        i = strformatdate(i)

    df = df.set_index('date')
    #df.index = df.index.astype('datetime64')
    df.to_csv(instrument + '_' + asi + '.csv', encoding='UTF8')