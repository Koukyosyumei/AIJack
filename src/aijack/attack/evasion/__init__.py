"""Subpackage for evasion attack, which creates a malicious data that the target
machine learning model cannot correctly classify.
"""
from .diva import DIVAWhiteBoxAttacker  # noqa: F401
from .evasion_attack import Evasion_attack_sklearn  # noqa: F401
from .fgsm import FGSMAttacker  # noqa: F401

__all__ = ["Evasion_attack_sklearn", "FGSMAttacker", "DIVAWhiteBoxAttacker"]
