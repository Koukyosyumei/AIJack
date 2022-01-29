import torch


class DataRepExtractor:
    def __init__(self, net, num_fc_layers=1, m=1, bias=True):
        self.net = net
        self.num_fc_layers = num_fc_layers
        self.m = m
        self.bias = bias

        # dl_dw = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
        # dl_dw = [g.detach() for g in dl_dw]
        # extractor = DataRepExtractor(net, num_fc_layers, m, bias)
        # datarep = extractor.extract_datarep(dl_dw)

    def get_dldw(self, loss):
        dldw = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True)
        dldw = [g.detach() for g in dldw]
        return dldw

    def extract_datarep(self, dldw):
        max_idx = torch.argmax(dldw[-2].norm(2, dim=1))
        rep_1 = dldw[-2][max_idx, :].reshape(1, -1)

        reps = [rep_1]
        for i in range(1, self.num_fc_layers):
            grad_idx = -2 * (i + 1) if self.bias else -2 * i + 1
            rep_i = (
                dldw[grad_idx][torch.topk(reps[-1].norm(2, dim=0), self.m)[1], :]
                .mean(dim=0)
                .reshape(1, -1)
            )
            reps.append(rep_i)

        return reps
