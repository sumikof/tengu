from datetime import datetime,timedelta

import pytz


def set_account(account):
    import configparser
    config = configparser.ConfigParser()
    config.read('oanda.conf')

    account.id =  config['OANDA']['accountID']
    account.token = config['OANDA']['access_token']



import matplotlib.pyplot as plt


def strformatdate(dateiso):
    iso = dateiso
    date = None
    try:
        date = datetime.strptime(iso[:-4], '%Y-%m-%dT%H:%M:%S.%f')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        print("formaterror", iso[:-4])

    return date


class act:

    def __init__(self):
        self.t = []
        self.y = []


    def init_graph(self):
        fig, self.ax = plt.subplots(1, 1)
        self.li, = self.ax.plot(self.t, self.y)


    def init_rate(self,rate):
        if "complete" in rate.keys() and rate["complete"] == True:
            self.t.append(strformatdate(rate["time"]))
            self.y.append(float(rate["mid"]["c"]))

    def rate_action(self,rate):

        def set(ar,val):
            ar.append(val)
            ar.pop(0)

        if "bids" in rate.keys() and "time" in rate.keys():
            time = strformatdate(rate["time"])
            bids = float(rate["bids"][0]["price"])
            set(self.t,time)
            set(self.y, bids)

        self.li.set_data(self.t,self.y)
        self.ax.set_xlim(max(self.t[-1] - timedelta(seconds=500),self.t[0]),self.t[-1])
        avg = sum(self.y)/len(self.y)
        haba = max(self.y) - min(self.y)
        print('avg = %f , haba = %f , time=%s ,bid=%f' % (avg,haba,self.t[-1],self.y[-1]))
        self.ax.set_ylim(avg-haba,avg+haba)
        #
        plt.pause(0.01)


def main():
    from oanda_api import oanda_stream_api,oanda_rest_api,Account
    account = Account()
    set_account(account)

    a = act()
    oanda_rest_api(account,instrument="USD_JPY",action=a.init_rate,count=100,granularity="S5")
    a.init_graph()
    oanda_stream_api(account,instrument="USD_JPY",action=a.rate_action)



if __name__ == '__main__':
    main()