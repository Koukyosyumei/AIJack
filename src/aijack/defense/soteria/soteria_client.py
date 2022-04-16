import torch

from ...manager import BaseManager


def attach_soteria_to_client(
    cls,
    input_layer,
    perturbed_layer,
    epsilon=0.2,
    target_layer_name=None,
):
    class SoteriaClientWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(SoteriaClientWrapper, self).__init__(*args, **kwargs)
            self.input_layer = input_layer
            self.perturbed_layer = perturbed_layer
            self.epsilon = epsilon
            self.target_layer_name = target_layer_name

            self._set_hook()

        def _set_hook(self):
            self.inputs = {}
            self.outputs = {}

            def get_input(name):
                def hook(model, x, output):
                    if not x[0].requires_grad:
                        raise ValueError("x.requires_grad should be True")
                    self.inputs[name] = x[0]

                return hook

            def get_output(name):
                def hook(model, x, output):
                    self.outputs[name] = output

                return hook

            getattr(self.model, self.input_layer).register_forward_hook(
                get_input(self.input_layer)
            )
            getattr(self.model, self.perturbed_layer).register_forward_hook(
                get_output(self.perturbed_layer)
            )

        def action_before_lossbackward(self):
            input_data = self.inputs[self.input_layer]
            feature = self.outputs[self.perturbed_layer]

            mask = torch.zeros_like(feature)
            r_dfr_dx_norm = torch.zeros_like(feature)

            rep_size = feature.shape[1]
            for i in range(rep_size):
                mask[:, i] = 1
                feature.backward(
                    mask, retain_graph=True
                )  # culc the derivative of feature_2 @ df_dtarget
                dfri_dx = input_data.grad.data
                r_dfr_dx_norm[:, i] = feature[:, i] / torch.norm(
                    dfri_dx.view(dfri_dx.shape[0], -1), dim=1
                )

                self.model.zero_grad()
                input_data.grad.data.zero_()
                mask[:, i] = 0

            self.topk_idxs = torch.topk(
                r_dfr_dx_norm.mean(dim=0), int(rep_size * self.epsilon)
            )[1]

        def action_after_lossbackward(self, target_layer_name=None):
            target_layer_name = (
                f"{self.perturbed_layer}.weight"
                if target_layer_name is None
                else target_layer_name
            )
            dl_dw = {
                layer_name: params.grad
                for layer_name, params in zip(
                    self.model.state_dict(), self.model.parameters()
                )
            }
            dl_dw[target_layer_name][self.topk_idxs, :] = 0

        def backward(self, loss):
            self.action_before_lossbackward()
            super().backward(loss)
            self.action_after_lossbackward(self.target_layer_name)

    return SoteriaClientWrapper


class SoteriaManager(BaseManager):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def attach(self, cls):
        return attach_soteria_to_client(cls, *self.args, **self.kwargs)
