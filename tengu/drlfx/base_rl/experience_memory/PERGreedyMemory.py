import heapq

from tengu.drlfx.base_rl.experience_memory.memory_abc import MemoryABC


class _head_wrapper():
    def __init__(self, data):
        self.data = data

    def __eq__(self, other):
        return True


class PERGreedyMemory(MemoryABC):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.max_priority = 1

    def add(self, experience):
        if self.capacity <= len(self.buffer):
            self.buffer.pop()

        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-self.max_priority, experience))

    def update(self, index, experience, td_error):
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-td_error, experience))

        if self.max_priority < td_error:
            self.max_priority = td_error

    def sample(self, batch_size):
        batchs = [heapq.heappop(self.buffer)[1].data for _ in range(batch_size)]
        return None , batchs

    def __len__(self):
        return len(self.buffer)

