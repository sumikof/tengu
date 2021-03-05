import random

class RateSeries:
    def __init__(self, rate_list, test_size ):
        self.rate_list = self._make_rate(rate_list)
        self.test_size = test_size
        self.start_pos = 0
        self.index = 0
        self.state_size = len(self._state())
        self.reset()

    @property
    def rate(self):
        return self._state()

    def reset(self):
        self.index = 0
        self.start_pos = random.randint(60*24, len(self.rate_list) - self.test_size)
        return self._is_done(), self.rate

    def next(self):
        self.index += 1
        return self._is_done(),self.rate

    def _make_rate(self,rate_list):
        import pandas as pd
        rates = pd.DataFrame(rate_list, columns=['1m'])
        rates['5m'] = rates['1m'].rolling(5).mean()
        rates['15m'] = rates['1m'].rolling(15).mean()
        rates['30m'] = rates['1m'].rolling(30).mean()
        rates['1h'] = rates['1m'].rolling(60).mean()
        rates['4h'] = rates['1m'].rolling(60 * 4).mean()
        rates['1d'] = rates['1m'].rolling(60 * 24).mean()
        return rates
    def _is_done(self):
        return self.test_size <= self.index

    def _state(self):
        return self.rate_list.iloc[self.start_pos + self.index].tolist()

    def __str__(self):
        return "RateList {}".format(self.rate_list)


if __name__ == '__main__':
    import pandas as pd
    rates = RateSeries(list(range(10000)),100)
    print(rates)