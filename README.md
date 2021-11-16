# secureml

This package implements papers about AI security such as Model Inversion, Poisoning Attack, Evasion Attack, Differential Privacy, and Homomorphic Encryption. We try to offer the same API for every paper to compare and combine different algorithms easily.

If you have any requests such as papers that you would like to see implemented, please raise an issue!

## Install

pip install git+https://github.com/Koukyosyumei/secure_ml

## Example

We are trying to provide all algorithms with the same API as much as possible.

```
# Model Inversion
attacker = MI_FACE(torch_model, ...)
reconstructed_image, _ = attacker.attack(target_label, learning_rate, num_iteration)
```

```
# Evasion Attack
attacker = Evasion_attack_sklearn(sklearn_classifier, ...)
generated_adversary_image, _ = attacker.attack(initial_image, ...)
```

```
# Poisoning Attack
attacker = Poison_attack_sklearn(sklearn_classifier, ...)
generated_adversary_image, _ = attacker.attack(initial_image, ...)
```

## Supported Papers

| Paper                                                                                                                                                                                                                                       | Type    | example                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------- |
| Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.                                                                               | Defense | Coming Soon!                                                                |
| Yang, Ziqi, et al. "Defending model inversion and membership inference attacks via prediction purification." arXiv preprint arXiv:2005.03915 (2020).                                                                                        | Defense | Coming Soon!                                                                |
| Shokri, Reza, et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017.                                                                                          | Attack  | [notebook](example/membership_inference/membership_inference_CIFAR10.ipynb) |  |
| Fredrikson, Matt, Somesh Jha, and Thomas Ristenpart. "Model inversion attacks that exploit confidence information and basic countermeasures." Proceedings of the 22nd ACM SIGSAC conference on computer and communications security. 2015.  | Attack  | [script](example/model_inversion/mi_face.py)                                |
| Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017. | Attack  | [script](example/model_inversion/gan_attack.py)                             |
| Biggio, Battista, et al. "Evasion attacks against machine learning at test time." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2013. attack                            | Attack  | [notebook](example/adversarial_example/example_evasion_attack_svm.ipynb)    |
| Biggio, Battista, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines." arXiv preprint arXiv:1206.6389 (2012).                                                                                              | Attack  | [notebook](example/adversarial_example/example_poison_attack.ipynb)         |
