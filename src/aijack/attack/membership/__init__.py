"""Subpackage for membership inference attack, which reveals the confidential information
    about whether the target data is in the training dataset or not.
"""

from .membership_inference import ShadowMembershipInferenceAttack  # noqa: F401

__all__ = ["ShadowMembershipInferenceAttack"]
