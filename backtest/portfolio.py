LONG = 1
SHORT = -1
OPEN = 0
CLOSE = 1


class Trade:
    def __init__(self, position_type, trade, date, rate, amount):
        self.position_type = position_type
        self.trade = trade
        self.date = date
        self.rate = rate
        self.amount = amount

    @property
    def position(self):
        return self.position_type

    @position.setter
    def position(self, value):
        self.__position = value

    def __str__(self):
        return "date %s, position: %s,trade: %s,rate %f, amount %d" % (
        self.date, self.position_type, self.trade, self.rate, self.amount)


class Position:
    position_type = 0
    rate = 0
    amount = 0

    def __init__(self, trade):
        self.position_type = trade.position_type
        self.rate = trade.rate
        self.amount = trade.amount

    def value(self, rate):
        return (rate - self.rate) * self.position_type * self.amount


class Portfolio:
    trading = []
    deals = None
    profit = 0
    balance = 0
    spread = 0

    def __init__(self, spread=0.0, deposit=0):
        self.spread = spread
        self.balance = deposit

    def deposit(self, dep):
        """
        入金
        :param equity:
        :return:
        """
        self.balance = dep

    def deal(self, date, position_type, rate, amount):
        """
        新規建て
        :param date:
        :param position_type:
        :param rate:
        :param amount:
        :return:
        """
        deal_rate = rate + (position_type * self.spread)
        trade = Trade(position_type, OPEN, date, deal_rate, amount)
        self.trading.append(trade)
        self.deals = Position(trade)
        #print("deal rate : {0:.3f} spread : {1:.3f}  position {2:.3f} ".format(rate,self.spread,deal_rate))

    def close_deal(self, date, rate, amount):
        """
        決済
        :param date:
        :param rate:
        :return:
        """
        trade = Trade(self.deals.position_type, CLOSE, date, rate, amount)
        self.trading.append(trade)
        position_rate = self.deals.rate
        profit = self.current_profit(rate)
        self.profit += profit
        self.balance += self.current_profit(rate)
        self.deals = None
        #print("close deal rate : {0:.3f} position : {0:.3f} profit : {0:.3f} ".format(rate,position_rate,profit))

    def position_rate(self):
        return self.deals.rate

    def current_profit(self, rate):
        if self.has_deals():
            return int(self.deals.value(rate))
        return 0

    def has_deals(self):
        return self.deals is not None


if __name__ == '__main__':
    portfolio = Portfolio(spread=0.018)
    portfolio.deposit(10000)

    portfolio.deal("2020/04/16", LONG, 110.165, 10000)
    portfolio.close_deal("2020/04/16", 110.123, 10000)

    portfolio.deal("2020/04/16", SHORT, 110.123, 10000)
    portfolio.close_deal("2020/04/16", 110.456, 10000)

    portfolio.deal("2020/04/16", SHORT, 113.21, 10000)
    portfolio.close_deal("2020/04/16", 110.9, 10000)

    portfolio.deal("2020/04/16", LONG, 113.5, 10000)
    portfolio.close_deal("2020/04/16", 110.1, 10000)

    portfolio.deal("2020/04/16", LONG, 113.415, 10000)

    print(portfolio.current_profit(113.415))
    print(portfolio.profit)
    for i in portfolio.trading:
        print(i)
