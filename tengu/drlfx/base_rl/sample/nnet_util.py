from .dueling_network import DuelingNNet
from .simple_nnet import SimpleNNet

from enum import Enum,auto

class NNetType(Enum):
    DuelingNNet = auto()
    SimpleNNet = auto()


def nnet_factory(nnet_type,learning_rate, num_status, num_actions, hidden_size):
    if nnet_type == NNetType.SimpleNNet:
        nnet = SimpleNNet(learning_rate, num_status, num_actions, hidden_size)
    elif nnet_type == NNetType.DuelingNNet:
        nnet = DuelingNNet(learning_rate, num_status, num_actions, hidden_size)
    else:
        raise RuntimeError

    return nnet
