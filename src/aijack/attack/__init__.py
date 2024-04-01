"""Submodule for attack algorithms against machine learning.
"""

from .base_attack import BaseAttacker  # noqa: F401
from .evasion import (  # noqa: F401
    DIVAWhiteBoxAttacker,
    Evasion_attack_sklearn,
    FGSMAttacker,
)
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
