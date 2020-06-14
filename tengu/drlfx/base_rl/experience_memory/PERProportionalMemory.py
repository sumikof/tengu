import random
import numpy as np

from tengu.drlfx.base_rl.experience_memory.memory_abc import MemoryABC


class _sum_tree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataidx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataidx]

    def __len__(self):
        return len(self.tree)


class PERProportionalMemory(MemoryABC):
    def __len__(self):
        return len(self.tree)

    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.tree = _sum_tree(capacity)
        self.alpha = alpha
        self.max_priority = 1

    def add(self, experience):
        """SumTreeに追加"""
        self.tree.add(self.max_priority, experience)

    def update(self, index, experience, td_error):
        """SumTreeの更新"""
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.update(index, priority)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size):
        """サンプリング"""
        indexes = []
        batchs = []  # 選ばれた経験
        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section * i + random.random() * section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)
        return indexes, batchs
