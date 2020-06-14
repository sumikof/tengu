import bisect
import random
import math

from tengu.drlfx.base_rl.experience_memory.memory_abc import MemoryABC


def rank_sum(k, a):
    return k * (2 + (k - 1) * a) / 2


def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a - 2 + math.sqrt((2 - a) ** 2 + 8 * a * k)
    return t / (2 * a)


class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0

    def __lt__(self, o):  # a<b
        return self.priority > o.priority


class PERRankBaseMemory(MemoryABC):
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha

        self.max_priority = 1

    def add(self, experience):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()

        experience = _bisect_wrapper(experience)
        experience.priority = self.max_priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001)  # priority を計算

        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size):
        indexes = []
        batchs = []

        # 合計値をだす
        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section * i + random.random() * section)

        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break
        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

        return indexes, batchs

    def __len__(self):
        return len(self.buffer)
