from tengu.backtest.portfolio import Portfolio
from tengu.backtest.rule import OrderRule


class LongOnlyFullOrder(OrderRule):

    def __init__(self,port):
        super(LongOnlyFullOrder,self).__init__(port)

    def deal_rule(self,rate):
        amount = 10000
        return amount

    def close_rule(self,rate):
        amount = 10000
        return amount



if __name__ == '__main__':
    import random

    branch = [True,False]

    # シミュレーションの試行回数
    for i in range(5):

        # 環境の初期化
        test_data = [100 + round(random.random(), 3) for i in range(30)]
        rule = LongOnlyFullOrder(Portfolio(spread=0.02,deposit=10000))

        # test_dataの数だけstep実行
        for rate in test_data:
            if rule.has_deals():
                if random.choice(branch):
                    rule.close_deal("", rate)
                else:
                    pass
            else:
                if random.choice(branch):
                    rule.deal("",rate)
                else:
                    pass
            if rule.portfolio.balance < 0:
                break
        print(rule.portfolio.balance)

