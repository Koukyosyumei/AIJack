from abc import ABCMeta, abstractmethod


class BaseManager(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def attach(self):
        pass
