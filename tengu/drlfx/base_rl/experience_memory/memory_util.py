from .PERProportionalMemory import PERProportionalMemory
from .PERRankBaseMemory import PERRankBaseMemory
from .PERGreedyMemory import PERGreedyMemory
from .ReplayMemory import ReplayMemory

from enum import Enum, auto


class MemoryType(Enum):
    PERProportionalMemory = auto()
    PERRankBaseMemory = auto()
    PERGreedyMemory = auto()
    ReplayMemory = auto()


def memory_factory(memory_type, memory_capacity, per_alpha=0.6):
    if memory_type == MemoryType.ReplayMemory:
        memory = ReplayMemory(memory_capacity)
    elif memory_type == MemoryType.PERGreedyMemory:
        memory = PERGreedyMemory(memory_capacity)
    elif memory_type == MemoryType.PERProportionalMemory:
        memory = PERProportionalMemory(memory_capacity, per_alpha)
    elif memory_type == MemoryType.PERRankBaseMemory:
        memory = PERRankBaseMemory(memory_capacity, per_alpha)
    else:
        raise RuntimeError

    return memory
