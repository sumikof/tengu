from tengu.backtest.portfolio import LONG


class RuleBase:
    def __init__(self, port):
        self.portfolio = port

    def deal_rule(self, rate):
        return 0

    def deal(self, date, rate):
        amount = self.deal_rule(rate)
        self.portfolio.deal(date=date, position_type=LONG, rate=rate, amount=amount)

    def close_rule(self, rate):
        return 0

    def close_deal(self, date, rate):
        amount = self.close_rule(rate)
        self.portfolio.close_deal(date=date, rate=rate, amount=amount)

    def current_profit(self):
        return self.portfolio.current_profit()

    def balance(self):
        return self.portfolio.balance

    def has_deals(self):
        return self.portfolio.has_deals()


class OrderRule(RuleBase):
    def __init__(self, port):
        super(OrderRule, self).__init__(port)

    def deal_rule(self, rate):
        return 0

    def close_rule(self, rate):
        return 0
