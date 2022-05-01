"""Subpackage for label leakage attack, which infere the private label information of the training dataset.
"""
from .normattack import NormAttackManager, attach_normattack_to_splitnn  # noqa: F401
