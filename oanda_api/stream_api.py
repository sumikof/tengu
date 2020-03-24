def oanda_stream_api(account, instrument,action):
    from oandapyV20 import API
    from oandapyV20.endpoints.pricing import PricingStream
    from oandapyV20.exceptions import V20Error

    params = {
        "instruments": instrument
    }
    api = API(access_token=account.token)
    ps = PricingStream(account.id, params)

    try:
        for rsp in api.request(ps):
            action(rsp)
    except V20Error as e:
        print("Error: {}".format(e))