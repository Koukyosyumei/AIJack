"""Subpackage for label leakage attack, which infere the private label information of the training dataset.
"""
from .normattack import (  # noqa: F401
    NormAttackSplitNNManager,
    attach_normattack_to_splitnn,
)

__all__ = ["NormAttackSplitNNManager", "attach_normattack_to_splitnn"]
