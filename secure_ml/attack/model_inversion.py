import torch


class Model_inversion:
    def __init__(self, torch_model, input_shape):
        self.model = torch_model
        self.input_shape = input_shape

    def attack(self, target_label,
               lam, num_itr, process_func=lambda x: x):
        """
        Args:
            target_label (int)
            lam (float) : step size
            num_itr (int) : number of iteration
            process_func (function) : default is identity function

        Returns:
            x_numpy (np.array) :
            loss ([float]) :
        """
        log = []
        x = torch.zeros(self.input_shape, requires_grad=True)
        for i in range(num_itr):
            c = process_func(1 - self.model(x)[:, [target_label]])
            c.backward()
            grad = x.grad
            with torch.no_grad():
                x -= lam * grad
            log.append(c.item())

        x_numpy = x.to('cpu').detach().numpy().copy()
        return x_numpy, log
