from abc import ABCMeta, abstractmethod


class NNetABC(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def train_on_batch(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, w):
        raise NotImplementedError

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError

    @abstractmethod
    def save_weights(self,file_name):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self,file_name):
        raise NotImplementedError