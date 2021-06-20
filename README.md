# secure_ml

This package allows you to experiment with various algorithms to attack and defense machine learning models.

## Install

pip install git+https://github.com/Koukyosyumei/secure_ml

## Supported Algorithms

The detailed explanations are available at the [example/README.md](example/README.md).

### Attack

1. Membership Inference
2. Model Inversion
3. Evasion Attack
4. Poisoning Attack

### Defense

1. DPSDG
2. Purification (WIP)

## Example notebooks

| algorithms           | example                                                                     | reference                                                            |
| -------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| DPSDG                | Coming Soon!                                                                | [original paper](https://arxiv.org/abs/1607.00133)                   |
| Purification         | Coming Soon!                                                                | [original paper](https://arxiv.org/abs/2005.03915)                   |
| membership inference | [notebook](example/membership_inference/membership_inference_CIFAR10.ipynb) | [original paper](https://arxiv.org/abs/1610.05820)                   |
| model inversion      | [notebook](example/model_inversion/example.ipynb)                           | [original paper](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677) |
| evasion attack       | [notebook](example/adversarial_example/example_evasion_attack_svm.ipynb)    | [original paper](https://arxiv.org/abs/1708.06131)                   |
| poisoning attack     | [notebook](example/adversarial_example/example_poison_attack.ipynb)         | [original paper](https://arxiv.org/abs/1206.6389)                    |
