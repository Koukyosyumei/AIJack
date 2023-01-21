from abc import ABCMeta, abstractmethod


class BaseManager(metaclass=ABCMeta):
    """Abstract class for Manager API"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def attach(self, cls):
        pass
