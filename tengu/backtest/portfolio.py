from logging import getLogger

logger = getLogger(__name__)

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
        self._position = None

    @property
    def position(self):
        return self.position_type

    @position.setter
    def position(self, value):
        self._position = value

    def __str__(self):
        return "(date %s, position: %s,trade: %s,rate %f, amount %d)" % (
            self.date, self.position_type, self.trade, self.rate, self.amount)

    def __repr__(self):
        return str(self)


class Position:
    position_type = 0
    rate = 0
    amount = 0

    def __init__(self, trade):
        self.position_type = trade.position_type
        self.rate = trade.rate
        self.amount = trade.amount

    def pl_rate(self, rate):
        return (rate - self.rate) * self.position_type

    def value(self, rate):
        return self.pl_rate(rate) * self.amount

    def __repr__(self):
        return "(position {},rate {},amount {})".format(self.position_type, self.rate, self.amount)


class Portfolio:

    def __init__(self, *, spread=0.0, deposit=0):
        self.spread = spread
        self._balance = deposit
        self.trading = []
        self.deals = None
        self.total_profit = 0

    def deposit(self, dep):
        """
        入金
        :param dep:
        :return:
        """
        self._balance = dep

    def reset(self, *, deposit=0):
        self.trading = []
        self.deals = None
        self.total_profit = 0
        self._balance = deposit

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
        logger.debug("deal rate : {0:.3f} spread : {1:.3f}  position {2:.3f} ".format(rate, self.spread, deal_rate))

    def close_deal(self, date, rate, amount):
        """
        決済
        :param date:
        :param rate:
        :param amount:
        :return:
        """
        position_rate = self.position_rate()
        trade = Trade(self.deals.position_type, CLOSE, date, rate, amount)
        self.trading.append(trade)
        profit = self.current_profit(rate)
        self.total_profit += profit
        self.balance += self.current_profit(rate)
        self.deals = None
        logger.debug(
            "close deal rate : {0:.3f} position : {0:.3f} profit : {0:.3f} ".format(rate, position_rate, profit))

    def position_rate(self):
        return self.deals.rate

    def has_deals(self):
        return self.deals is not None

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, new_balance):
        self._balance = new_balance

    def current_profit(self, rate):
        if self.has_deals():
            return int(self.deals.value(rate))
        return 0

    def pl_rate(self, rate):
        if self.has_deals():
            return self.deals.pl_rate(rate)
        return 0

    def current_balance(self, rate):
        return self.balance + self.current_profit(rate)

    def __str__(self):
        status = """
        spread = {}
        balance = {}
        trading = {}
        deals = {} 
        """
        return status.format(self.spread, self.balance, self.trading, self.deals)


if __name__ == '__main__':
    portfolio = Portfolio(spread=0)
    portfolio.deposit(10000)

    portfolio.deal("2020/04/16", LONG, 107.720, 8000)
    profit = portfolio.current_profit(108.432)
    print(profit)  # 0.007
    name="hoge"
    age = 10
    def logform(**kwargs):
        return kwargs
    print(logform(name=name,age=age))
