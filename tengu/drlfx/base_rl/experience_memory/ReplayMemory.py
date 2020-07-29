import random

from tengu.drlfx.base_rl.experience_memory.memory_abc import MemoryABC


class ReplayMemory(MemoryABC):
    def __init__(self, memory_capacity):
        self.capacity = memory_capacity  # メモリの最大
        self.memory = []
        self.index = 0

    @classmethod
    def build(cls, builder):
        return ReplayMemory(builder.args.get("memory_capacity", 10000))

    def add(self, obj):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = obj

        # indexを１つずらす、capacityを超えたら0に戻る
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return None, random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def update(self, index, experience, td_error):
        pass