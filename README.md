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

<img src="logo/AIJACK-NEON-LOGO.png" width=400 align="right">

AIJack allows you to assess the privacy and security risks of machine learning algorithms such as *Model Inversion*, *Poisoning Attack*, *Evasion Attack*, *Free Rider*, and *Backdoor Attack*. AIJack also provides various defense techniques like *Differential Privacy*, *Homomorphic Encryption*, and other heuristic approaches. In addition, AIJack provides APIs for many distributed learning schemes like *Federated Learning* and *Split Learning*. You can integrate many attack and defense methods into such collaborative learning with a few lines. We currently implement more than 30 state-of-arts methods. For more information, see the [documentation](https://koukyosyumei.github.io/AIJack/intro.html).

# Installation

You can install AIJack with `pip`. AIJack requires Boost and pybind11.

```
apt install -y libboost-all-dev
pip install -U pip
pip install "pybind11[global]"

pip install git+https://github.com/Koukyosyumei/AIJack
```

You can also use our [Dockerfile](Dockerfile).


# Quick Start

We briefly introduce some example usages. You can also find more examples in [documentation](https://koukyosyumei.github.io/AIJack/intro.html).

## Basic Interface

For standard machine learning algorithm, AIJack allows you to simulate attacks against machine learning models with `Attacker` APIs. AIJack mostly supports PyTorch or sklearn models.

abstract code
```Python
attacker = Attacker(target_model)
result = attacker.attack()
```

For distributed learning such as Fedeated Learning, AIJack offers four basic APIs: `Client`, `Server`, `API`, and `Manager`. `Client` and `Server` represents each client and server within each distributed learning scheme, and we register the clients and servers to `API`. You can run this `API` and execute training via `run` method. `Manager` gives additional abilities such as attack, defense or parallel computing to `Client`, `Server` or `API` via `attach` method.

abstract code
```Python
client = [Client(), Client()]
server = Server()
api = API(client, server)
api.run() # execute training

c_manager = ClientManager()
s_manager = ServerManager()
ExtendedClient = c_manager.attach(Client)
ExtendedServer = c_manager.attach(Server)

extended_client = [ExtendedClient(), ExtendedClient()]
extended_server = ExtendedServer()
api = API(extended_client, extended_server)
api.run() # execute training
```

## Federated Learning

### FedAVG

FedAVG is the most representative algorithm of Federated Learning, where multiple clients jointly train a single model without sharing their local datasets. You can integrate any Pytorch models.

```Python
from aijack.collaborative.fedavg import FedAVGClient, FedAVGServer

clients = [FedAVGClient(local_model_1, user_id=0), FedAVGClient(local_model_2, user_id=1)]
optimizers = [optim.SGD(clients[0].parameters()), optim.SGD(clients[1].parameters())]

server = FedAVGServer(clients, global_model)

api = FedAVGAPI(
    server,
    clients,
    criterion,
    optimizers,
    dataloaders
)
api.run()
```

### FedMD

Model-Distillation based Federated Learning does not need communicating gradients, which might decrease the information leakage.

```Python
from aijack.collaborative.fedmd import FedMDAPI, FedMDClient, FedMDServer

clients = [
    FedMDClient(Net().to(device), public_dataloader, output_dim=10, user_id=c)
    for c in range(client_size)
]
local_optimizers = [optim.SGD(client.parameters(), lr=lr) for client in clients]

server = FedMDServer(clients, Net().to(device))

api = FedMDAPI(
    server,
    clients,
    public_dataloader,
    local_dataloaders,
    F.nll_loss,
    local_optimizers,
    test_dataloader,
    num_communication=2,
)
api.run()
```

### SecureBoost (Vertical Federated version of XGBoost)

AIJack supports not only neuralnetwork but also tree-based Federated Learning.

```Python
from aijacl.collaborative.tree import SecureBoostClassifierAPI, SecureBoostClient

keygenerator = PaillierKeyGenerator(512)
pk, sk = keygenerator.generate_keypair()

sclf = SecureBoostClassifierAPI(2,subsample_cols,min_child_weight,depth,min_leaf,
                  learning_rate,boosting_rounds,lam,gamma,eps,0,0,1.0,1,True)

sp1 = SecureBoostClient(x1, 2, [0], 0, min_leaf, subsample_cols, 256, False, 0)
sp2 = SecureBoostClient(x2, 2, [1], 1, min_leaf, subsample_cols, 256, False, 0)
sparties = [sp1, sp2]

sparties[0].set_publickey(pk)
sparties[0].set_secretkey(sk)
sparties[1].set_publickey(pk)

sclf.fit(sparties, y)
sclf.predict_proba(X)
```

### MPI-backend

AIJack supports MPI-backend for some of Federated Learning methods.

FedAVG
```Python
from mpi4py import MPI
from aijack.collaborative.fedavg import FedAVGClient, FedAVGServer
from aijack.collaborative.fedavg import MPIFedAVGAPI, MPIFedAVGClientManager, MPIFedAVGServerManager

comm = MPI.COMM_WORLD
myid = comm.Get_rank()

mpi_client_manager = MPIFedAVGClientManager()
mpi_server_manager = MPIFedAVGServerManager()
MPIFedAVGClient = mpi_client_manager.attach(FedAVGClient)
MPIFedAVGServer = mpi_server_manager.attach(FedAVGServer)

if myid == 0:
    server = MPIFedAVGServer(comm, FedAVGServer(client_ids, model))
    api = MPIFedAVGAPI(
        comm,
        server,
        True,
        F.nll_loss,
        None,
        None,
        num_rounds,
        1,
    )
else:
    client = MPIFedAVGClient(comm, FedAVGClient(model, user_id=myid))
    api = MPIFedAVGAPI(
        comm,
        client,
        False,
        F.nll_loss,
        optimizer,
        dataloader,
        num_rounds,
        1,
    )

api.run()
```

FedMD
```Python
from mpi4py import MPI
from aijack.collaborative.fedmd import MPIFedMDAPI, MPIFedMDClient, MPIFedMDServer

comm = MPI.COMM_WORLD
myid = comm.Get_rank()

if myid == 0:
    server = MPIFedMDServer(comm, FedMDServer(client_ids, model))
    api = MPIFedMDAPI(
        comm,
        server,
        True,
        F.nll_loss,
        None,
        None,
    )
else:
    client = MPIFedMDClient(comm, FedMDClient(model, public_dataloader, output_dim=10, user_id=myid))
    api = MPIFedMDAPI(
        comm,
        client,
        False,
        F.nll_loss,
        optimizer,
        dataloader,
        public_dataloader,
    )

api.run()
```

### Attack: Model Inversion

Model Inversion Attack steals the local training data via the shared information like gradients or parameters.

```Python
from aijack.attack.inversion import GradientInversionAttackServerManager

manager = GradientInversionAttackServerManager(input_shape, distancename="l2")
GradientInversionAttackFedAVGServer = manager.attach(FedAVGServer)

server = GradientInversionAttackFedAVGServer(clients, global_model)

api = FedAVGAPI(
    server,
    clients,
    criterion,
    optimizers,
    dataloaders
)
api.run()

reconstructed_training_data = server.attack()
```

### Defense: Differential Privacy

One possible defense against Model Inversion Attack is using differential privacy. AIJack supports DPSGD, an optimizer which makes the trained model satisfy differential privacy.

```Python
from aijack.defense.dp import DPSGDManager, GeneralMomentAccountant, DPSGDClientManager

dp_accountant = GeneralMomentAccountant()
dp_manager = DPSGDManager(
    accountant,
    optim.SGD,
    dataset=trainset,
)

manager = DPSGDClientManager(dp_manager)
DPSGDFedAVGClient = manager.attach(FedAVGClient)

clients = [DPSGDFedAVGClient(local_model_1, user_id=0), DPSGDFedAVGClient(local_model_2, user_id=1)]
```

### Defense: Soteria

Another defense algorithm soteria, which theoretically gurantees the lowerbound of reconstructino error.

```Python
from aijack.defense.soteria import SoteriaClientManager

manager = SoteriaClientManager("conv", "lin", target_layer_name="lin.0.weight")
SoteriaFedAVGClient = manager.attach(FedAVGClient)

clients = [SoteriaFedAVGClient(local_model_1, user_id=0), SoteriaFedAVGClient(local_model_2, user_id=1)]
```

### Defense: Homomorophic Encryption

Clients in Federated Learning can also encrypt their local gradients to prevent the potential information leakage. For example, AIJack offers Paiilier Encryption with c++ backend, which faster than other python-based implementations.

```Python
from aijack.defense.paillier import PaillierGradientClientManager, PaillierKeyGenerator

keygenerator = PaillierKeyGenerator(key_length)
pk, sk = keygenerator.generate_keypair()

manager = PaillierGradientClientManager(pk, sk)
PaillierGradFedAVGClient = manager.attach(FedAVGClient)

clients = [
  PaillierGradFedAVGClient(local_model_1, user_id=0, server_side_update=False),
  PaillierGradFedAVGClient(local_model_2, user_id=1, server_side_update=False)
    ]

server = FedAVGServer(clients, global_model, lr=lr, server_side_update=False)
```

### Attack: Poisoning

Poisoning Attack aims to deteriorate the performance of the trained model.

One famous approach is Label Flip Attack.

```Python
from aijack.attack.poison import LabelFlipAttackClientManager

manager = LabelFlipAttackClientManager(victim_label=0, target_label=1)
LabelFlipAttackFedAVGClient = manager.attach(FedAVGClient)

clients = [LabelFlipAttackFedAVGClient(local_model_1, user_id=0), FedAVGClient(local_model_2, user_id=1)]
```

### Defense: FoolsGOld

One of the standard method to mitigate Poisoning Attack is FoolsGold, which calculates the similarity among clients and decrease the influence of the malicious clients.

```Python
from aijack.defense.foolsgold import FoolsGoldServerManager

manager = FoolsGoldServerManager()
FoolsGoldFedAVGServer = manager.attach(FedAVGServer)
server = FoolsGoldFedAVGServer(clients, global_model)
```

### Attack: FreeRider

In real situation where the center server pay money for clients, it is important to detect freeriders who do not anything but pretend to locally train their models.

```Python
from aijack.attack.freerider import FreeRiderClientManager

manager = FreeRiderClientManager(mu=0, sigma=1.0)
FreeRiderFedAVGClient = manager.attach(FedAVGClient)

clients = [FreeRiderFedAVGClient(local_model_1, user_id=0), FedAVGClient(local_model_2, user_id=1)]
```

## Split Learning

Split Learning is another collaborative learning scheme, where only one party owns the ground-truth labels.

### SplitNN

```Python
from aijack.collaborative.splitnn import SplitNNAPI, SplitNNClient

clients = [SplitNNClient(model_1, user_id=0), SplitNNClient(model_2, user_id=1)]
optimizers = [optim.Adam(model_1.parameters()), optim.Adam(model_2.parameters())]

splitnn = SplitNNAPI(clients, optimizers, train_loader, criterion, num_epoch)
splitnn.run()
```

### Attack: Label Leakage

AIJack supports norm-based label leakage attack against Split Learning.

```Python
from aijack.attack.labelleakage import NormAttackManager

manager = NormAttackManager(criterion, device="cpu")
NormAttackSplitNNAPI = manager.attach(SplitNNAPI)
normattacksplitnn = NormAttackSplitNNAPI(clients, optimizers)
leak_auc = normattacksplitnn.attack(target_dataloader)
```


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

|                          | Attack Type          | Example                                                   | Paper                                                                                                                                               |
| ------------------------ | -------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| MI-FACE                  | Model Inversion      | [example](docs/aijack_miface.ipynb)                       | [paper](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677)                                                                                         |
| DLG                      | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb)    | [paper](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html)                                                      |
| iDLG                     | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb)    | [paper](https://arxiv.org/abs/2001.02610)                                                                                                           |
| GS                       | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb)    | [paper](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)                                              |
| CPL                      | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb)    | [paper](https://arxiv.org/abs/2004.10397)                                                                                                           |
| GradInversion            | Model Inversion      | [example](docs/aijack_gradient_inversion_attack.ipynb)    | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf) |
| GAN Attack               | Model Inversion      | [example](example/model_inversion/gan_attack.py)          | [paper](https://arxiv.org/abs/1702.07464)                                                                                                           |
| Shadow Attack            | Membership Inference | [example](docs/aijack_membership_inference.ipynb)         | [paper](https://arxiv.org/abs/1610.05820)                                                                                                           |
| Norm attack              | Label Leakage        | [example](docs/aijack_split_learning.ipynb)               | [paper](https://arxiv.org/abs/2102.08504)                                                                                                           |
| Delta Weights            | Free Rider Attack    | WIP                                                       | [paper](https://arxiv.org/pdf/1911.12560.pdf)                                                                                                       |
| Gradient descent attacks | Evasion Attack       | [example](docs/aijack_evasion_attack.ipynb)               | [paper](https://arxiv.org/abs/1708.06131)                                                                                                           |
| DBA                      | Backdoor Attack      | WIP                                                       | [paper](https://openreview.net/forum?id=rkgyS0VFvr)                                                                                                 |
| Label Flip Attack        | Poisoning Attack     | [example](docs/aijack_poisoning_federated_learning.ipynb) | [paper](https://arxiv.org/abs/2203.08669)                                                                                                           |
| History Attack           | Poisoning Attack     | [example](docs/aijack_poisoning_federated_learning.ipynb) | [paper](https://arxiv.org/abs/2203.08669)                                                                                                           |
| MAPF                     | Poisoning Attack     | [example](docs/aijack_poisoning_federated_learning.ipynb) | [paper](https://arxiv.org/abs/2203.08669)                                                                                                           |
| SVM Poisoning            | Poisoning Attack     | [example](docs/aijack_poisoning_attack_svm.ipynb)         | [paper](https://arxiv.org/abs/1206.6389)                                                                                                            |


## Defense

|                 | Defense Type           | Example                                  | Paper                                                                                                                                                              |
| --------------- | ---------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DPSGD           | Differential Privacy   | [example](docs/aijack_miface.ipynb)      | [paper](https://arxiv.org/abs/1607.00133)                                                                                                                          |
| Paillier        | Homomorphic Encryption | [example](docs/aijack_secureboost.ipynb) | [paper](https://link.springer.com/chapter/10.1007/3-540-48910-X_16)                                                                                                |  |
| CKKS            | Homomorphic Encryption | [test](test/defense/ckks/test_core.py)   | [paper](https://eprint.iacr.org/2016/421.pdf)                                                                                                                      |  |
| Soteria         | Others                 | [example](docs/aijack_soteria.ipynb)     | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.pdf) |
| FoolsGold       | Others                 | WIP                                      | [paper](https://arxiv.org/abs/1808.04866)                                                                                                                          |
| Sparse Gradient | Others                 | [example](docs/aijack_fedavg.ipynb)      | [paper](https://aclanthology.org/D17-1045/)                                                                                                                        |
| MID             | Others                 | [example](docs/aijack_mid.ipynb)         | [paper](https://arxiv.org/abs/2009.05241)                                                                                                                          |

-----------------------------------------------------------------------

# Contact

welcome2aijack[@]gmail.com
