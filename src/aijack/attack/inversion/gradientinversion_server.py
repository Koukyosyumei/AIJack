from ...manager import BaseManager
from .gradientinversion import GradientInversion_Attack


def _default_gradinent_inversion_attack_on_receive(self):
    tmp_result = []
    for s in range(self.num_trial_per_communication):
        self.reset_seed(s)
        try:
            tmp_result.append(self.attack())
        except:
            continue
    self.attack_results.append(tmp_result)


def attach_gradient_inversion_attack_to_server(
    cls,
    x_shape,
    attack_function_on_receive=_default_gradinent_inversion_attack_on_receive,
    num_trial_per_communication=1,
    target_client_id=0,
    **gradinvattack_kwargs,
):
    """Wraps the given class in GradientInversionServerWrapper.

    Args:
        x_shape: input shape of target_model.
        attack_function_on_receive (function, optional): a function to execute attack called after receving the local gradients. Defaults to _default_gradinent_inversion_attack_on_receive.
        num_trial_per_communication (int, optional): number of attack trials executed per communication. Defaults to 1.
        target_client_id (int, optional): id of target client. Default to 0.
        gradinvattack_kwargs: kwargs for GradientInversion_Attack

    Returns:
        cls: GradientInversionServerWrapper
    """

    class GradientInversionServerWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(GradientInversionServerWrapper, self).__init__(*args, **kwargs)
            self.target_client_id = target_client_id
            self.num_trial_per_communication = num_trial_per_communication
            self.attacker = GradientInversion_Attack(
                self.server_model, x_shape, **gradinvattack_kwargs
            )

            self.attack_results = []

        def change_target_client_id(self, target_client_id):
            self.target_client_id = target_client_id
            self.attacker.target_model = self.clients[self.target_client_id]

        def receive(self, *args, **kwargs):
            super(GradientInversionServerWrapper, self).receive(*args, **kwargs)

            attack_function_on_receive(self)

        def attack(self, **kwargs):
            received_gradient = self.uploaded_gradients[self.target_client_id]
            return self.attacker.attack(received_gradient, **kwargs)

        def group_attack(self, **kwargs):
            received_gradient = self.uploaded_gradients[self.target_client_id]
            return self.attacker.group_attack(received_gradient, **kwargs)

        def reset_seed(self, seed):
            self.attacker.reset_seed(seed)

    return GradientInversionServerWrapper


class GradientInversionAttackServerManager(BaseManager):
    """Manager class for Gradient-based model inversion attack"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        """Wraps the given class in GradientInversionServerWrapper.

        Returns:
            cls: GradientInversionServerWrapper
        """
        return attach_gradient_inversion_attack_to_server(
            cls, *self.args, **self.kwargs
        )
