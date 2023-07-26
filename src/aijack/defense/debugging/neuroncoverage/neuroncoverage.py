import copy

import torch
import torch.nn as nn


def scale(out, dim=-1, rmax=1, rmin=0):
    output_std = (out - out.min()) / (out.max() - out.min())
    output_scaled = output_std * (rmax - rmin) + rmin
    return output_scaled


def step_through_model(model, prefix=""):
    for name, module in model.named_children():
        path = "{}/{}".format(prefix, name)
        if (
            isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
        ):  # test for dataset
            yield (path, name, module)
        else:
            yield from step_through_model(module, path)


def get_model_layers(model, cross_section_size=0):
    layer_dict = {}
    i = 0
    for path, name, module in step_through_model(model):
        layer_dict[str(i) + path] = module
        i += 1
    if cross_section_size > 0:
        target_layers = list(layer_dict)[0::cross_section_size]
        layer_dict = {
            target_layer: layer_dict[target_layer] for target_layer in target_layers
        }
    return layer_dict


def get_layer_output_sizes(model, layer_dict, data):
    output_sizes = {}
    hooks = []

    def hook(module, input, output):
        module_idx = len(output_sizes)
        m_key = list(layer_dict)[module_idx]
        output_sizes[m_key] = list(output.size()[1:])

    for name, module in layer_dict.items():
        hooks.append(module.register_forward_hook(hook))

    try:
        model(data[:1])
    finally:
        for h in hooks:
            h.remove()

    return output_sizes


def get_intermediate_outputs(model, data, module, force_relu=True):
    outputs = []

    def hook(module, input, output):
        if force_relu:
            outputs.append(torch.relu(output))
        else:
            outputs.append(output)

    handle = module.register_forward_hook(hook)
    model(data)
    handle.remove()
    return torch.stack(outputs)


class NeuronCoverageTracker:
    def __init__(self, model, dummy_data, threshold=0.9, device="cpu"):
        self.model = model
        self.threshold = threshold
        self.device = device
        self.layer_dict = get_model_layers(model)
        self.output_shape_of_layers = get_layer_output_sizes(
            model, self.layer_dict, dummy_data
        )
        self.cov_tracker = {}

    def _init_cov_tracker(self):
        for layer_name, layer_shape in self.output_shape_of_layers.items():
            self.cov_tracker[layer_name] = (
                torch.zeros(layer_shape[0]).type(torch.BoolTensor).to(self.device)
            )

    def _update_cov_tracker(self, x):
        for (
            layer_name,
            layer_module,
        ) in self.layer_dict.items():
            layer_intermediate_output = get_intermediate_outputs(
                self.model, x, layer_module
            )
            layer_intermediate_output = torch.squeeze(
                torch.sum(layer_intermediate_output, dim=1)
            )
            layer_intermediate_output_scaled = scale(layer_intermediate_output)
            threshold = self.threshold
            mask_index = layer_intermediate_output_scaled > threshold
            self.cov_tracker[layer_name] = mask_index | self.cov_tracker[layer_name]

    def coverage(self, dataloader, pos_of_x=0, initialize=True, update=True):
        if initialize:
            self._init_cov_tracker()
        if not update:
            cov_tracker_prev = copy.deepcopy(self.cov_tracker)

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

        if not update:
            self.cov_tracker = cov_tracker_prev

        return (num_covered_neurons / num_total_neurons).item()
