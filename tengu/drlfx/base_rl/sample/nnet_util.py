from .dueling_network import DuelingNNet
from .simple_nnet import SimpleNNet

from enum import Enum,auto

class NNetType(Enum):
    DuelingNNet = auto()
    SimpleNNet = auto()


def nnet_factory(builder):
    if builder.args["nnet_type"] == NNetType.SimpleNNet.name:
        nnet = SimpleNNet.build(builder)
    elif builder.args["nnet_type"] == NNetType.DuelingNNet.name:
        nnet = DuelingNNet.build(builder)
    else:
        raise RuntimeError

    return nnet
