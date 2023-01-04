<!--
  Title: AIJack
  Description: AIJack is a fantastic framework demonstrating the security risks of machine learning and deep learning, such as Model Inversion, poisoning attack, and membership inference attack.
  Author: Hideaki Takahashi
  -->

# AIJack: Security and Privacy Risk Simulator for Standard/Distributed Machine Learning

<div align="left">
<img src="https://badgen.net/github/stars/Koukyosyumei/AIjack?color=green">
<img src="https://badgen.net/github/forks/Koukyosyumei/AIjack">
<img src="https://badgen.net/github/watchers/Koukyosyumei/AIjack">
<img src="https://img.shields.io/github/commit-activity/m/Koukyosyumei/AIJack">
<img src="https://img.shields.io/github/languages/code-size/Koukyosyumei/AIJack">
<img src="https://img.shields.io/github/languages/count/Koukyosyumei/AIJack">
<img src="https://img.shields.io/github/license/Koukyosyumei/AIJack">
</div>

❤️ <i>If you like AIJack, please consider <a href="https://github.com/sponsors/Koukyosyumei">becoming a GitHub Sponsor</a></i> ❤️

# What is AIJack?

<img src="logo/AIJACK-NEON-LOGO.png" width=406 align="right">

AIJack allows you to assess the privacy and security risks of machine learning algorithms such as *Model Inversion*, *Poisoning Attack*, *Evasion Attack*, *Free Rider*, and *Backdoor Attack*. AIJack also provides various defense techniques like *Differential Privacy*, *Homomorphic Encryption*, and other heuristic approaches. In addition, AIJack provides APIs for many distributed learning schemes like *Federated Learning* and *Split Learning*. You can integrate many attack and defense methods into such collaborative learning with a few lines. We currently implement more than 30 state-of-arts methods. For more information, see the [documentation](https://koukyosyumei.github.io/AIJack/intro.html).

# Installation

You can install AIJack with `pip`. AIJack requires Boost and pybind11.

```
apt install -y libboost-all-dev
pip install -U pip
pip install "pybind11[global]"

pip install aijack
```

If you want to use the latest-version, you can directly install from GitHub.

```
pip install git+https://github.com/Koukyosyumei/AIJack
```

We also provide [Dockerfile](Dockerfile).


# Quick Start

We briefly introduce the overview of AIJack.

## Features

- Flexible API for more than 30 attack and defense alorithms
- Compatible with PyTorch and scikit-learn
- Support for both Deep Learning and classical ML
- Fast Implementation with C++ backend
- Pythorch-Extension for Homomorphic Encryption
- MPI-Backend for Federated Learning

## Basic Interface

For standard machine learning algorithms, AIJack allows you to simulate attacks against machine learning models with `Attacker` APIs. AIJack mainly supports PyTorch or sklearn models.

```Python
# abstract code

attacker = Attacker(target_model)
result = attacker.attack()
```

For distributed learning such as Federated Learning and Split Learning, AIJack offers four basic APIs: `Client`, `Server`, `API`, and `Manager`. `Client` and `Server` represent each client and server within each distributed learning scheme. You can execute training by registering the clients and servers to `API` and running it. `Manager` gives additional abilities such as attack, defense, or parallel computing to `Client`, `Server` or `API` via `attach` method.

```Python
# abstract code

client = [Client(), Client()]
server = Server()
api = API(client, server)
api.run() # execute training

c_manager = ClientManagerForAdditionalAbility(...)
s_manager = ServerManagerForAdditionalAbility(...)
ExtendedClient = c_manager.attach(Client)
ExtendedServer = c_manager.attach(Server)

extended_client = [ExtendedClient(...), ExtendedClient(...)]
extended_server = ExtendedServer(...)
api = API(extended_client, extended_server)
api.run() # execute training
```

For example, the bellow code implements the scenario, where the server in Federated Learning tries to steal the training data with model inversion attack, and one client aims to defense this attack with differential privacy.

```Python
from aijack.collaborative.fedavg import FedAVGClient, FedAVGServer, FedAVGAPI
from aijack.defense.dp import DPSGDClientManager

manager = DPSGDClientManager(...)
DPSGDFedAVGClient = manager.attach(FedAVGClient)

manager = GradientInversionAttackServerManager(...)
GradientInversionAttackFedAVGServer = manager.attach(FedAVGServer)


clients = [FedAVGClient(...), DPSGDFedAVGClient(...)]
server = GradientInversionAttackFedAVGServer(...)

api = API(extended_client, extended_server)
api.run()
```

## Resources

You can also find more examples in our tutorials and documentation.

- Tutorials
- [Documentation](https://koukyosyumei.github.io/AIJack/intro.html)

# Supported Algorithms

|               |                        |                                                                                                                                                                                                                                |
| ------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Collaborative | Horizontal FL          | [FedAVG](https://arxiv.org/abs/1602.05629),[FedProx]((https://arxiv.org/abs/1812.06127)) ,[FedMD]((https://arxiv.org/abs/2108.13323)),[FedGEMS]((https://arxiv.org/abs/2110.11027)),[DSFL]((https://arxiv.org/abs/2008.06180)) |
| Collaborative | Vertical FL            | [SplitNN](https://arxiv.org/abs/1812.00564),[SecureBoost](https://arxiv.org/abs/1901.08755)                                                                                                                                    |
| Attack        | Model Inversion        | MI-FACE, DLG, iDLG, GS, CPL, GradInversion, GAN Attack                                                                                                                                                                         |
| Attack        | Label Leakage          | Norm Attack                                                                                                                                                                                                                    |
| Attack        | Poisoning              | History Attack, Label Flip, MAPF, SVM Poisoning                                                                                                                                                                                |
| Attack        | Backdoor               | DBA                                                                                                                                                                                                                            |
| Attack        | Free-Rider             | Delta-Weight                                                                                                                                                                                                                   |
| Attack        | Evasion                | Gradient-Descent Attack                                                                                                                                                                                                        |
| Attack        | Membership Inference   | Shaddow Attack                                                                                                                                                                                                                 |
| Defense       | Homomorphic Encryption | Paiilier, CKKS                                                                                                                                                                                                                 |
| Defense       | Differential Privacy   | Moment Accountant, DPSGD                                                                                                                                                                                                       |
| Defense       | Others                 | Soteria, FoolsGold, MID,Sparse Gradient                                                                                                                                                                                        |

-----------------------------------------------------------------------

# Contact

welcome2aijack[@]gmail.com
