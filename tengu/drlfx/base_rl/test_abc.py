from abc import ABCMeta, abstractmethod


class TestABC(metaclass=ABCMeta):

    @property
    @abstractmethod
    def mask(self):
        return [True]

    @abstractmethod
    def reset(self):
        state = None
        return state

    @abstractmethod
    def step(self, action, environment):
        next_state, reward, done, info = None, None, None, None
        return next_state, reward, done, info

    @abstractmethod
    def is_finish(self):
        return False

    @abstractmethod
    def check_status_is_done(self, state):
        return False
