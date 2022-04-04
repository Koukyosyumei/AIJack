import torch
from matplotlib import pyplot as plt

from ..base_attack import BaseAttacker


class MI_FACE(BaseAttacker):
    """Implementation of model inversion attack
    reference: https://dl.acm.org/doi/pdf/10.1145/2810103.2813677

    Attributes:
        target_model: model of the victim
        input_shape: input shapes of taregt model
        auxterm_func (function): the default is constant function
        process_func (function): the default is identity function
    """

    def __init__(
        self,
        target_model,
        input_shape=(1, 1, 64, 64),
        target_label=0,
        lam=0.01,
        num_itr=100,
        auxterm_func=lambda x: 0,
        process_func=lambda x: x,
        apply_softmax=False,
        device="cpu",
        log_interval=1,
        log_show_img=False,
        show_img_func=lambda x: x * 0.5 + 0.5,
    ):
        """Inits MI_FACE
        Args:
            target_model: model of the victim
            input_shape: input shapes of taregt model
            auxterm_func (function): the default is constant function
            process_func (function): the default is identity function
        """
        super().__init__(target_model)
        self.input_shape = input_shape
        self.target_label = target_label
        self.lam = lam
        self.num_itr = num_itr

        self.auxterm_func = auxterm_func
        self.process_func = process_func
        self.device = device
        self.log_interval = log_interval
        self.log_show_img = log_show_img
        self.apply_softmax = apply_softmax
        self.show_img_func = show_img_func

        self.log_image = []

    def attack(
        self,
        init_x=None,
    ):
        """Execute the model inversion attack on the target model.

        Args:
            target_label (int): taregt label
            lam (float): step size
            num_itr (int): number of iteration

        Returns:
            best_img: inversed image with the best score
            log :
        """
        log = []
        if init_x is None:
            x = torch.zeros(self.input_shape, requires_grad=True).to(self.device)
        else:
            init_x = init_x.to(self.device)
            x = init_x

        best_score = float("inf")
        best_img = None

        for i in range(self.num_itr):
            x = x.detach()
            x.requires_grad = True
            pred = self.target_model(x)
            pred = pred.softmax(dim=1) if self.apply_softmax else pred
            target_pred = pred[:, [self.target_label]]
            c = 1 - target_pred + self.auxterm_func(x)
            c.backward()
            grad = x.grad

            if c.item() < best_score:
                best_img = x

            with torch.no_grad():
                x -= self.lam * grad
                x = self.process_func(x)
            log.append(c.item())

            if self.log_interval != 0 and i % self.log_interval == 0:
                print(f"epoch {i}: {c.item()}")
                self._show_img(x)

            self.log_image.append(x.clone())

        self._show_img(x)

        return best_img, log

    def _show_img(self, x):
        if self.log_show_img:
            if self.input_shape[1] == 1:
                plt.imshow(
                    self.show_img_func(x.detach().cpu().numpy()[0][0]),
                    cmap="gray",
                )
                plt.show()
            else:
                plt.imshow(
                    self.show_img_func(x.detach().cpu().numpy()[0].transpose(1, 2, 0))
                )
                plt.show()
