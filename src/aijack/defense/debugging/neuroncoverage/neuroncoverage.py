import torch


class NeuronCoverageTracker:
    def __init__(self, model, threshold, device):
        self.model = model
        self.threshold = threshold
        self.device = device
        self.output_shape_of_layers = {}
        self.cov_tracker = {}

    def _init_cov_tracker(self):
        for layer_name, layer_shape in self.output_shape_of_layers.items():
            self.cov_tracker[layer_name] = (
                torch.zeros(layer_shape[0]).type(
                    torch.BoolTensor).to(self.device)
            )

    def _update_cov_tracker(self, x):
        intermediate_outputs_of_layer = {}
        for (
            layer_name,
            layer_intermediate_output,
        ) in intermediate_outputs_of_layer.items():
            layer_intermediate_output_scaled = None
            mask_index = layer_intermediate_output_scaled > self.threshold
            is_covered = mask_index.sum(0) > 0
            self.cov_tracker[layer_name] = is_covered | self.cov_tracker[layer_name]

    def coverage(self, dataloader, pos_of_x=0, initialize=True):
        if initialize:
            self._init_cov_tracker()

        for data in dataloader:
            if pos_of_x is None:
                x = data.to(self.decice)
            else:
                x = data[pos_of_x].to(self.device)
            self._update_cov_tracker(x)

        num_covered_neurons = 0
        num_total_neurons = 0
        for is_covered in self.cov_tracker.values():
            num_covered_neurons += is_covered.sum()
            num_total_neurons += len(is_covered)

        return (num_covered_neurons / num_total_neurons).item()
