import random

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大
        self.memory = []
        self.index = 0

    def add(self, obj):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = obj

        # indexを１つずらす、capacityを超えたら0に戻る
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
