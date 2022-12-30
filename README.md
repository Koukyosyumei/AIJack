<!--
  Title: AIJack
  Description: AIJack is a fantastic framework demonstrating the security risks of machine learning and deep learning, such as Model Inversion, poisoning attack, and membership inference attack.
  Author: Hideaki Takahashi
  -->

<h1 align="center">

  <br>
  <img src="logo/logo_small.png" alt="AIJack" width="200"></a>
  <br>
  Let's hijack AI!
  <br>

</h1>

<div align="center">
<img src="https://badgen.net/github/watchers/Koukyosyumei/AIjack">
<img src="https://badgen.net/github/stars/Koukyosyumei/AIjack?color=green">
<img src="https://badgen.net/github/forks/Koukyosyumei/AIjack">
</div>

<div align="center">
❤️ <i>If you like AIJack, please consider <a href="https://github.com/sponsors/Koukyosyumei">becoming a GitHub Sponsor</a></i> ❤️
</div>

# What is AIJack?

AIJack allows you to assess the privacy and security risks of machine learning algorithms such as *Model Inversion*, *Poisoning Attack* and *Evasion Attack*. AIJack also provides various defense techniques like *Federated Learning*, *Split Learning*, *Differential Privacy*, *Homomorphic Encryption*, and other heuristic approaches. We currently implement more than 20 state-of-arts methods. We also support MPI for some of the distributed algorithms. For more information, see the [documentation](https://koukyosyumei.github.io/AIJack/intro.html).

# Table of Contents


- [What is AIJack?](#what-is-aijack)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [pip](#pip)
  - [Docker](#docker)
- [Supported Algorithms](#supported-algorithms)
  - [Distributed Learning](#distributed-learning)
  - [Attack](#attack)
  - [Defense](#defense)
- [Quick Start](#quick-start)
  - [Federated Learning and Model Inversion Attack](#federated-learning-and-model-inversion-attack)
  - [Split Learning and Label Leakage Attack](#split-learning-and-label-leakage-attack)
  - [DPSGD (SGD with Differential Privacy)](#dpsgd-sgd-with-differential-privacy)
  - [Federated Learning with Homomorphic Encryption](#federated-learning-with-homomorphic-encryption)
  - [SecureBoost (XGBoost with Homomorphic Encryption)](#secureboost-xgboost-with-homomorphic-encryption)
  - [Evasion Attack](#evasion-attack)
  - [Poisoning Attack](#poisoning-attack)
- [Contact](#contact)

# Installation

## pip

AIJack requires Boost and pybind11.

```
apt install -y libboost-all-dev
pip install -U pip
pip install "pybind11[global]"

pip install git+https://github.com/Koukyosyumei/AIJack
```

## Docker

Please use our [Dockerfile](Dockerfile).

# Supported Algorithms

## Distributed Learning

|             | Example                                           | Paper                                     |
| ----------- | ------------------------------------------------- | ----------------------------------------- |
| FedAVG      | [example](docs/aijack_fedavg.ipynb)               | [paper](https://arxiv.org/abs/1602.05629) |
| FedProx     | WIP                                               | [paper](https://arxiv.org/abs/1812.06127) |
| FedKD       | [example](test/collaborative/fedkd/test_fedkd.py) | [paper](https://arxiv.org/abs/2108.13323) |
| FedMD       | [example](docs/aijack_fedmd.ipynb)                | [paper](https://arxiv.org/abs/1910.03581) |
| FedGEMS     | WIP                                               | [paper](https://arxiv.org/abs/2110.11027) |
| DSFL        | WIP                                               | [paper](https://arxiv.org/abs/2008.06180) |
| SplitNN     | [example](docs/aijack_split_learning.ipynb)       | [paper](https://arxiv.org/abs/1812.00564) |
| SecureBoost | [example](docs/aijack_secureboost.ipynb)          | [paper](https://arxiv.org/abs/1901.08755) |

## Attack

|                          | Attack Type          | Example                                                | Paper                                                                                                                                               |
| ------------------------ | -------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| MI-FACE                  | Model Inversion      | [example](docs/aijack_miface.ipynb)                    | [paper](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677)                                                                                         |
| DLG                      | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb) | [paper](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)                                                      |
| iDLG                     | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb) | [paper](https://arxiv.org/abs/2001.02610)                                                                                                           |
| GS                       | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb) | [paper](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)                                              |
| CPL                      | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb) | [paper](https://arxiv.org/abs/2004.10397)                                                                                                           |
| GradInversion            | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb) | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf) |
| GAN Attack               | Model Inversion      | [example](example/model_inversion/gan_attack.py)       | [paper](https://arxiv.org/abs/1702.07464)                                                                                                           |
| Shadow Attack            | Membership Inference | [example](docs/aijack_membership_inference.ipynb)      | [paper](https://arxiv.org/abs/1610.05820)                                                                                                           |
| Norm attack              | Label Leakage        | [example](docs/aijack_split_learning.ipynb)            | [paper](https://arxiv.org/abs/2102.08504)                                                                                                           |
| Delta Weights            | Free Rider Attack    | WIP                                                    | [paper](https://arxiv.org/pdf/1911.12560.pdf)                                                                                                       |
| Gradient descent attacks | Evasion Attack       | [example](docs/aijack_evasion_attack.ipynb)            | [paper](https://arxiv.org/abs/1708.06131)                                                                                                           |
| SVM Poisoning            | Poisoning Attack     | [example](docs/aijack_poison_attack.ipynb)             | [paper](https://arxiv.org/abs/1206.6389)                                                                                                            |


## Defense

|          | Defense Type           | Example                                  | Paper                                                                                                                                                              |
| -------- | ---------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DPSGD    | Differential Privacy   | [example](docs/aijack_miface.ipynb)      | [paper](https://arxiv.org/abs/1607.00133)                                                                                                                          |
| Paillier | Homomorphic Encryption | [example](docs/aijack_secureboost.ipynb) | [paper](https://link.springer.com/chapter/10.1007/3-540-48910-X_16)                                                                                                |  |
| CKKS     | Homomorphic Encryption | [test](test/defense/ckks/test_core.py)   | [paper](https://eprint.iacr.org/2016/421.pdf)                                                                                                                      |  |
| Soteria  | Others                 | [example](docs/aijack_soteria.ipynb)     | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.pdf) |
| MID      | Others                 | [example](docs/aijack_mid.ipynb)         | [paper](https://arxiv.org/abs/2009.05241)                                                                                                                          |


# Quick Start

We briefly introduce some example usages. You can also find more examples in [`example`](example).

## Federated Learning and Model Inversion Attack

FedAVG is the most representative algorithm of Federated Learning, where multiple clients jointly train a single model without sharing their local datasets.

- Base

You can write the process of FedAVG like the standard training with Pytorch.

```Python
from aijack.collaborative import FedAvgClient, FedAvgServer

clients = [FedAvgClient(local_model_1, user_id=0), FedAvgClient(local_model_2, user_id=1)]
optimizers = [optim.SGD(clients[0].parameters()), optim.SGD(clients[1].parameters())]
server = FedAvgServer(clients, global_model)

for client, local_trainloader, local_optimizer in zip(clients, trainloaders, optimizers):
    for data in local_trainloader:
        inputs, labels = data
        local_optimizer.zero_grad()
        outputs = client(inputs)
        loss = criterion(outputs, labels.to(torch.int64))
        client.backward(loss)
        optimizer.step()
server.action()
```

- Attack

You can simulate the model inversion attack against FedAVG.

```Python
from aijack.attack import GradientInversion_Attack

dlg_manager = GradientInversionAttackManager(input_shape, distancename="l2")
FedAvgServer_DLG = dlg.attach(FedAvgServer)
server = FedAvgServer_DLG(clients, global_model, lr=lr)

reconstructed_image, reconstructed_label = server.attack()
```

- Defense

One possible defense for clients of FedAVG is Soteria, and you need only two additional lines to implement Soteria.

```Python
from aijack.collaborative import FedAvgClient
from aijack.defense import SoteriaManager

manager = SoteriaManager("conv", "lin", target_layer_name="lin.0.weight")
SoteriaFedAvgClient = manager.attach(FedAvgClient)
client = SoteriaFedAvgClient(Net(), user_id=i, lr=lr)
```

## Split Learning and Label Leakage Attack

You can use split learning, where only one party has the ground-truth labels.

- Base

```Python
from aijack.collaborative import SplitNN, SplitNNClient

clients = [SplitNNClient(model_1, user_id=0), SplitNNClient(model_2, user_id=1)]
optimizers = [optim.Adam(model_1.parameters()), optim.Adam(model_2.parameters())]
splitnn = SplitNN(clients, optimizers)

for data dataloader:
    splitnn.zero_grad()
    inputs, labels = data
    outputs = splitnn(inputs)
    loss = criterion(outputs, labels)
    splitnn.backward(loss)
    splitnn.step()
```

- Attack

We support norm-based label leakage attack against Split Learning.

```Python
from aijack.attack import NormAttackManager
from aijack.collaborative import SplitNN

manager = NormAttackManager(criterion, device="cpu")
NormAttackSplitNN = manager.attach(SplitNN)
normattacksplitnn = NormAttackSplitNN(clients, optimizers)
leak_auc = normattacksplitnn.attack(target_dataloader)
```

## DPSGD (SGD with Differential Privacy)

DPSGD is an optimizer based on Differential Privacy and theoretically privatizes your deep learning model. We implement the core of differential privacy mechanisms with C++, which is faster than many other libraries purely implemented with Python.

```Python
from aijack.defense import GeneralMomentAccountant
from aijack.defense import PrivacyManager

accountant = GeneralMomentAccountant(noise_type="Gaussian", search="greedy", orders=list(range(2, 64)), bound_type="rdp_tight_upperbound")
privacy_manager = PrivacyManager(accountant, optim.SGD, l2_norm_clip=l2_norm_clip, dataset=trainset, iterations=iterations)
dpoptimizer_cls, lot_loader, batch_loader = privacy_manager.privatize(noise_multiplier=sigma)

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

## Federated Learning with Homomorphic Encryption

```Python
  from aijack.collaborative import FedAvgClient, FedAvgServer
  from aijack.defense import PaillierGradientClientManager, PaillierKeyGenerator

keygenerator = PaillierKeyGenerator(64)
pk, sk = keygenerator.generate_keypair()

manager = PaillierGradientClientManager(pk, sk)
PaillierGradFedAvgClient = manager.attach(FedAvgClient)

clients = [
    PaillierGradFedAvgClient(Net(), user_id=i, lr=lr, server_side_update=False)
    for i in range(client_num)
]
```

## SecureBoost (XGBoost with Homomorphic Encryption)

SecureBoost is a vertically federated version of XGBoost, where each party encrypts sensitive information with Paillier Encryption. You need additional compile to use secureboost, which requires Boost 1.65 or later.

```
cd src/aijack/collaborative/tree
pip install -e .
```


```Python
from aijack_secureboost import SecureBoostParty, SecureBoostClassifier, PaillierKeyGenerator

keygenerator = PaillierKeyGenerator(512)
pk, sk = keygenerator.generate_keypair()

sclf = SecureBoostClassifier(2,subsample_cols,min_child_weight,depth,min_leaf,
                  learning_rate,boosting_rounds,lam,gamma,eps,0,0,1.0,1,True)

sp1 = SecureBoostParty(x1, 2, [0], 0, min_leaf, subsample_cols, 256, False, 0)
sp2 = SecureBoostParty(x2, 2, [1], 1, min_leaf, subsample_cols, 256, False, 0)

sparties = [sp1, sp2]

sparties[0].set_publickey(pk)
sparties[1].set_publickey(pk)
sparties[0].set_secretkey(sk)

sclf.fit(sparties, y)

sclf.predict_proba(X)

```

## Evasion Attack

Evasion Attack generates data that the victim model cannot classify correctly.

```Python
from aijack.attack import Evasion_attack_sklearn

attacker = Evasion_attack_sklearn(target_model=clf, X_minus_1=attackers_dataset)
result, log = attacker.attack(initial_datapoint)
```

## Poisoning Attack

Poisoning Attack injects malicious data into the training dataset to control the behavior of the trained models.

```Python
from aijack.attack import Poison_attack_sklearn

attacker = Poison_attack_sklearn(clf, X_train_, y_train_, t=0.5)
xc_attacked, log = attacker.attack(xc, 1, X_valid, y_valid)
```

-----------------------------------------------------------------------

# Contact

welcome2aijack[@]gmail.com
