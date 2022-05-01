"""Subpackage for poisoning attack, which inserts malicious data to the training dataset,
so that the performance of the trained machine learning model will degregate.
"""
from .poison_attack import Poison_attack_sklearn  # noqa: F401
