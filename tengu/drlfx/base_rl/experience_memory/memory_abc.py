from abc import ABCMeta, abstractmethod


class MemoryABC(metaclass=ABCMeta):
    @abstractmethod
    def add(self, obj):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, index, experience, td_error):
        raise NotImplementedError

    def __bool__(self):
        return True