"""Subpackage for poisoning attack, which inserts malicious data to the training dataset,
so that the performance of the trained machine learning model will degregate.
"""
from .label_flip import LabelFlipAttackManager  # noqa: F401
from .poison_attack import Poison_attack_sklearn  # noqa: F401
