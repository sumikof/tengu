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



def memory_factory(memory_type,**kwargs):
    memory_capacity = 'memory_capacity'
    per_alpha = 'per_alpha'
    if memory_type == MemoryType.ReplayMemory.name:
        memory = ReplayMemory(
            kwargs.get(memory_capacity,10000))
    elif memory_type == MemoryType.PERGreedyMemory.name:
        memory = PERGreedyMemory(
            kwargs.get(memory_capacity,10000))
    elif memory_type == MemoryType.PERProportionalMemory.name:
        memory = PERProportionalMemory(
            kwargs.get(memory_capacity,10000),
            kwargs.get(per_alpha, 0.6))
    elif memory_type == MemoryType.PERRankBaseMemory.name:
        memory = PERRankBaseMemory(
            kwargs.get(memory_capacity,10000),
            kwargs.get(per_alpha, 0.6))
    else:
        raise RuntimeError

    return memory
