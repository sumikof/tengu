import heapq


class _head_wrapper():
    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        return True


class PERGreedyMemory:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.max_priority = 1

    def add(self, experience):
        if self.capacity <= len(self.buffer):
            self.buffer.pop()

        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-self.max_priority, experience))

    def update(self, experience, td_error):
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-td_error, experience))

        if self.max_priority < td_error:
            self.max_priority = td_error

    def sample(self, batch_size):
        batchs = [heapq.heappop(self.buffer)[1].d for _ in range(batch_size)]
        return batchs
