from abc import ABCMeta, abstractmethod


class BaseAttacker(metaclass=ABCMeta):
    def __init__(self, target_model):
        """attacker against ml model

        Args:
            target_model: target ML model
        """
        self.target_model = target_model

    @abstractmethod
    def attack(self):
        pass
