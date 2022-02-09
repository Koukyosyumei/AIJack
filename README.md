<!--
  Title: AIJack
  Description: AIJack is a fantastic framework to demonstrate security risks of machine learning and deep learning, such as model inversion attack, poisoning attack, and membership inference attack.
  Author: Hideaki Takahashi
  -->

<h1 align="center">

  <br>
  <img src="logo/logo_small.png" alt="AIJack" width="200"></a>
  <br>
  Try to hijack AI!
  <br>

</h1>

<div align="center">
<img src="https://badgen.net/github/watchers/Koukyosyumei/AIjack">
<img src="https://badgen.net/github/stars/Koukyosyumei/AIjack?color=green">
<img src="https://badgen.net/github/forks/Koukyosyumei/AIjack">
</div>

# AIJack

This package implements algorithms for AI security such as Model Inversion, Poisoning Attack, Evasion Attack, Differential Privacy, and Homomorphic Encryption. For example, you can experiment with a variant gradient inversion attack (a kind of model inversion attack) with the same API.

```Python
# DLG Attack (Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).)
attacker = GradientInversion_Attack(net, input_shape, distancename="l2")

# GS Attack (Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." Advances in Neural Information Processing Systems 33 (2020): 16937-16947.)
attacker = GradientInversion_Attack(net, input_shape, distancename="cossim", tv_reg_coef=0.01)

# iDLG (Zhao, Bo, Konda Reddy Mopuri, and Hakan Bilen. "idlg: Improved deep leakage from gradients." arXiv preprint arXiv:2001.02610 (2020).)
attacker = GradientInversion_Attack(net, input_shape, distancename="l2", optimize_label=False)

# CPL (Wei, Wenqi, et al. "A framework for evaluating gradient leakage attacks in federated learning." arXiv preprint arXiv:2004.10397 (2020).)
attacker = GradientInversion_Attack(net, input_shape, distancename="l2", optimize_label=False,
                                        lm_reg_coef=0.01)

# GradInversion (Yin, Hongxu, et al. "See through gradients: Image batch recovery via gradinversion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.)
attacker = GradientInversion_Attack(net, input_shape,
                                                  distancename="l2", optimize_label=False,
                                                  bn_reg_layers=[net.body[1], net.body[4], net.body[7]],
                                                  group_num = 5,
                                                  tv_reg_coef=0.00, l2_reg_coef=0.0001,
                                                  bn_reg_coef=0.001, gc_reg_coef=0.001)
                                                  
received_gradients = torch.autograd.grad(loss, net.parameters())
received_gradients = [cg.detach() for cg in received_gradients]
attacker.attack(received_gradients)
```


## Contents

- [AIJack](#aijack)
  - [Contents](#contents)
  - [Install](#install)
  - [Supported Techniques](#supported-techniques)
    - [Collaborative Learning](#collaborative-learning)
    - [Attack](#attack)
    - [Defense](#defense)
  - [Usage](#usage)
    - [Collaborative Learning](#collaborative-learning-1)
    - [Attack](#attack-1)
    - [Defense](#defense-1)
  - [Supported Papers](#supported-papers)

## Install

```
pip install git+https://github.com/Koukyosyumei/AIJack
```

## Supported Techniques

### Collaborative Learning

- [Federated Learning](example/collaborative_learning/README.md)
- [Split Learning](example/collaborative_learning/README.md)

### Attack

- [evasion attack](example/adversarial_example/example_evasion_attack_svm.ipynb)
- [poisoning attack](example/adversarial_example/example_poison_attack.ipynb)
- [model inversion attack (simple pytorch model)](example/model_inversion/mi_face.py)
- [model inversion attack (split learning)](example/model_inversion/generator_attack.py)
- [model inverison attack (gradient inversion)](example/model_inversion/gradient_inversion_attack.md)
- [membership inference attack](example/membership_inference/membership_inference_CIFAR10.ipynb)
- [label leakage attack with split learning](example/label_leakage/label_leakage.py)

### Defense

- [Differential Privacy](example/differential_privacy/README.md)
- [Soteria](example/model_inversion/soteria.py)
- [POC of Homomorphic Encryption](test/defense/ckks/test_core.py)

## Usage

### Collaborative Learning

<details><summary>Federated Learning</summary><div>

```python
clients = [TorchModule(), TorcnModule()]
global_model = TorchModule()
server = FedAvgServer(clients, global_model)

for _ in range(epoch):

  for client in clients:
    normal pytorch training.

  server.update()
  server.distribtue()
```
</div></details>

<details><summary>Split Learning</summary><div>

```python
client_1 = SplitNNClient(first_model, user_id=0)
client_2 = SplitNNClient(second_model, user_id=1)
clients = [client_1, client_2]
splitnn = SplitNN(clients)

for _ in range(epoch):
  for x, y in dataloader:

    for opt in optimizers:
      opt.zero_grad()

    pred = splitnn(x)
    loss = criterion(y, pred)
    loss.backwad()
    splitnn.backward()

    for opt in optimizers:
      opt.step()
```
</div></details>

### Attack

<details><summary>Evasion Attack</summary><div>

```python
attacker = Evasion_attack_sklearn(
    target_model=clf,
    X_minus_1=attackers_dataset,
    dmax=(5000 / 255) * 2.5,
    max_iter=300,
    gamma=1 / (X_train.shape[1] * np.var(X_train)),
    lam=10,
    t=0.5,
    h=10,
)

result, log = attacker.attack(initial_datapoint)
```

</div></details>

<details><summary>Poisonning Attack</summary><div>

```python
attacker = Poison_attack_sklearn(clf, X_train_, y_train_, t=0.5)
xc_attacked, log = attacker.attack(xc, 1, X_valid, y_valid_, num_iterations=200)
```

</div></details>

### Defense

<details><summary>Moment Accountant</summary><div>

```Python
ga = GeneralMomentAccountant(noise_type="Gaussian",
                             search="greedy",
                             precision=0.001,
                             orders=list(range(2, 64)),
                             bound_type="rdp_tight_upperbound")
ga.add_step_info({"sigma":noise_multiplier}, sampling_rate, iterations)
ga.get_epsilon(delta)
```

</div></details>

<details><summary>DPSGD</summary><div>

```Python
privacy_manager = PrivacyManager(
        accountant,
        optim.SGD,
        l2_norm_clip=l2_norm_clip,
        dataset=trainset,
        lot_size=lot_size,
        batch_size=batch_size,
        iterations=iterations,
    )

dpoptimizer_cls, lot_loader, batch_loader = privacy_manager.privatize(
        noise_multiplier=sigma
    )

for data in lot_loader(trainset):
    X_lot, y_lot = data
    optimizer.zero_grad()
    for X_batch, y_batch in batch_loader(TensorDataset(X_lot, y_lot)):
        optimizer.zero_grad_keep_accum_grads()
        pred = net(X_batch)
        loss = criterion(pred, y_batch.to(torch.int64))
        loss.backward()
        optimizer.update_accum_grads()
    optimizer.step()
```

</div></details>

<details><summary>Soteria</summary><div>

```Python
client = SetoriaFedAvgClient(Net(), "conv", "lin", user_id=i, lr=lr)

normal fedavg training

client.action_before_lossbackward()
loss.backward()
client.action_after_lossbackward("lin.0.weight")
```

</div></details>

## Supported Papers

| Paper                                                                                                                                                                                                                                       | Type    | example                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------------------- |
| Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.                                                                               | Defense | [script](example/model_inversion/mi_face_differential_privacy.py)           |
| Yang, Ziqi, et al. "Defending model inversion and membership inference attacks via prediction purification." arXiv preprint arXiv:2005.03915 (2020).                                                                                        | Defense | Coming Soon!                                                                |
| Shokri, Reza, et al. "Membership inference attacks against machine learning models." 2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017.                                                                                          | Attack  | [notebook](example/membership_inference/membership_inference_CIFAR10.ipynb) |  |
| Fredrikson, Matt, Somesh Jha, and Thomas Ristenpart. "Model inversion attacks that exploit confidence information and basic countermeasures." Proceedings of the 22nd ACM SIGSAC conference on computer and communications security. 2015.  | Attack  | [script](example/model_inversion/mi_face.py)                                |
| Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017. | Attack  | [script](example/model_inversion/gan_attack.py)                             |
| Biggio, Battista, et al. "Evasion attacks against machine learning at test time." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2013. attack                            | Attack  | [notebook](example/adversarial_example/example_evasion_attack_svm.ipynb)    |
| Biggio, Battista, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines." arXiv preprint arXiv:1206.6389 (2012).                                                                                              | Attack  | [notebook](example/adversarial_example/example_poison_attack.ipynb)         |
| Li, Oscar, et al. "Label leakage and protection in two-party split learning." arXiv preprint arXiv:2102.08504 (2021).                                                                                                                       | Attack  | [script](example/label_leakage/label_leakage.py)                            |
| Geiping, Jonas, et al. "Inverting Gradients--How easy is it to break privacy in federated learning?." arXiv preprint arXiv:2003.14053 (2020).                                                                                               | Attack  | [script](example/model_inversion/dlg_gs.py)                                 |
| Zhu, Ligeng, and Song Han. "Deep leakage from gradients." Federated learning. Springer, Cham, 2020. 17-31.                                                                                                                                  | Attack  | [script](example/model_invresion/../model_inversion/dlg_gs.py)              |
| Sun, Jingwei, et al. "Soteria: Provable Defense Against Privacy Leakage in Federated Learning From Representation Perspective." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.                    | Defense | [script](example/model_inversion/fedavg_dlg_gs.py)                          |
