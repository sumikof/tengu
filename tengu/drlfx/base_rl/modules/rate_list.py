import random
import copy

class RateList:
    def __init__(self, rate_list, state_size, test_size ):
        self.rate_list = rate_list
        self.state_size = state_size
        self.test_size = test_size
        self._mini_batch = None
        self.index = 0
        self.reset()

    def copy_rate(self):
        return copy.copy(self._state())

    @property
    def rate(self):
        return self._state()

    def reset(self):
        self.index = 0
        random_index = random.randint(0, len(self.rate_list) - self.test_size)
        self._mini_batch = self.rate_list[random_index:random_index + self.test_size]
        return self._is_done(), self.rate

    def next(self):
        self.index += 1
        return self._is_done(),self.rate

    def _is_done(self):
        return self.test_size - self.state_size < self.index

    def _state(self):
        return self._mini_batch[self.index : self.index + self.state_size]


if __name__ == '__main__':
    rates = RateList(list(range(100)),3,10)
    print(rates._mini_batch)
    print(rates.rate)
    done,_ = rates.next()
    while(not done):
        print(rates.rate)
        print(rates.rate[-1])
        print("-----")
        done,_ = rates.next()
