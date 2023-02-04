"""Submodule for attack algorithms against machine learning.
"""
from .base_attack import BaseAttacker  # noqa: F401
from .evasion import Evasion_attack_sklearn  # noqa: F401
from .inversion import (  # noqa: F401
    MI_FACE,
    GANAttackClientManager,
    Generator_Attack,
    GradientInversion_Attack,
    GradientInversionAttackServerManager,
    attach_ganattack_to_client,
)
from .labelleakage import (  # noqa: F401
    NormAttackSplitNNManager,
    attach_normattack_to_splitnn,
)
from .membership import ShadowMembershipInferenceAttack  # noqa: F401
from .poison import Poison_attack_sklearn  # noqa: F401

__all__ = [
    "BaseAttacker",
    "Evasion_attack_sklearn",
    "MI_FACE",
    "GANAttackClientManager",
    "Generator_Attack",
    "GradientInversion_Attack",
    "GradientInversionAttackServerManager",
    "attach_ganattack_to_client",
    "NormAttackSplitNNManager",
    "attach_normattack_to_splitnn",
    "Poison_attack_sklearn",
    "ShadowMembershipInferenceAttack",
]
