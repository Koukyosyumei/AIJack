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

This package implements algorithms for AI security such as Model Inversion, Poisoning Attack, Evasion Attack, Differential Privacy, and Homomorphic Encryption.

## Install

```
# pip install pybind11 (uncomment if necessary)
pip install git+https://github.com/Koukyosyumei/AIJack
```

## Usage

### Collaborative Learning

- FedAVG

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

- SplitNN

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

### Attack against Federated Learning

- Gradient Inversion (server-side model inversion attack against federated learning)

```Python
from aijack.attack import GradientInversion_Attack

# DLG Attack (Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).)
dlg_manager = GradientInversionAttackManager(input_shape, distancename="l2")
FedAvgServer_DLG = dlg.attach(FedAvgServer)

"""
# GS Attack (Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." Advances in Neural Information Processing Systems 33 (2020): 16937-16947.)
gs_manager = GradientInversionAttackManager(input_shape, distancename="cossim", tv_reg_coef=0.01)
FedAvgServer_GS = gs.attach(FedAvgServer)

# iDLG (Zhao, Bo, Konda Reddy Mopuri, and Hakan Bilen. "idlg: Improved deep leakage from gradients." arXiv preprint arXiv:2001.02610 (2020).)
idlg_manager = GradientInversionAttackManager(input_shape, distancename="l2", optimize_label=False)
FedAvgServer_iDLG = idlg.attach(FedAvgServer)

# CPL (Wei, Wenqi, et al. "A framework for evaluating gradient leakage attacks in federated learning." arXiv preprint arXiv:2004.10397 (2020).)
cpl_manager = GradientInversionAttackManager(input_shape, distancename="l2", optimize_label=False, lm_reg_coef=0.01)
FedAvgServer_CPL = cpl.attach(FedAvgServer)

# GradInversion (Yin, Hongxu, et al. "See through gradients: Image batch recovery via gradinversion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.)
gi_manager = GradientInversionAttackManager(input_shape, distancename="l2", optimize_label=False, bn_reg_layers=[net.body[1], net.body[4], net.body[7]],
                                    group_num = 5, tv_reg_coef=0.00, l2_reg_coef=0.0001, bn_reg_coef=0.001, gc_reg_coef=0.001)
FedAvgServer_GI = gi.attach(FedAvgServer)
"""

server = FedAvgServer_DLG(clients, global_model, lr=lr)
# --- normal federated learning --- #
reconstructed_image, reconstructed_label = server.attack()
```

- GAN Attack (client-side model inversion attack against federated learning)

```Python
# Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the # 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017.
from aijack.attack import GANAttackManager
from aijack.collaborative import FedAvgClient

manager = GANAttackManager(
    target_label,
    generator,
    optimizer_g,
    criterion,
    nz=nz,
)
GANAttackFedAvgClient = manager.attach(FedAvgClient)
client = GANAttackFedAvgClient(client)
# --- normal federated learning --- #
reconstructed_image = client.attack(1)
```

### Defense for Federated Learning

- Soteria

```Python
# Sun, Jingwei, et al. "Soteria: Provable defense against privacy leakage in federated learning from representation perspective." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
from aijack.collaborative import FedAvgClient
from aijack.defense import SoteriaManager

manager = SoteriaManager("conv", "lin", target_layer_name="lin.0.weight")
SoteriaFedAvgClient = manager.attach(FedAvgClient)
client = SoteriaFedAvgClient(Net(), user_id=i, lr=lr)
# --- normal FL training ---
```

### Attack against Split Learning

- Label Leakage Attack

```Python
# Li, Oscar, et al. "Label leakage and protection in two-party split learning." arXiv preprint arXiv:2102.08504 (2021).
from aijack.attack import NormAttackManager
from aijack.collaborative import SplitNN

manager = NormAttackManager(criterion, device="cpu")
NormAttackSplitNN = manager.attach(SplitNN)
normattacksplitnn = NormAttackSplitNN(clients, optimizers)
# --- normal split learning --- #
leak_auc = normattacksplitnn.attack(target_dataloader)
```

### Other Attacks

- MI-FACE (model inversion attack)

```Python
# Fredrikson, Matt, Somesh Jha, and Thomas Ristenpart. "Model inversion attacks that exploit confidence information and basic countermeasures." Proceedings of the 22nd # ACM SIGSAC conference on computer and communications security. 2015.
from aijack.attack import MI_FACE

mi = MI_FACE(target_torch_net, input_shape)
reconstructed_data, _ = mi.attack(target_label, lam, num_itr)
```

- Evasion Attack

```python
# Biggio, Battista, et al. "Evasion attacks against machine learning at test time." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2013.
from aijack.attack import Evasion_attack_sklearn

attacker = Evasion_attack_sklearn(target_model=clf, X_minus_1=attackers_dataset)
result, log = attacker.attack(initial_datapoint)
```

- Poisoning Attack

```python
# Biggio, Battista, Blaine Nelson, and Pavel Laskov. "Poisoning attacks against support vector machines." arXiv preprint arXiv:1206.6389 (2012).
from aijack.attack import Poison_attack_sklearn

attacker = Poison_attack_sklearn(clf, X_train_, y_train_, t=0.5)
xc_attacked, log = attacker.attack(xc, 1, X_valid, y_valid)
```

### Other Defences


- DPSGD (Differential Privacy)

```Python
#  Abadi, Martin, et al. "Deep learning with differential privacy." Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.
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

- MID (Defense against model inversion attak)

```Python
# Wang, Tianhao, Yuheng Zhang, and Ruoxi Jia. "Improving robustness to model inversion attacks via mutual information regularization." arXiv preprint arXiv:2009.05241 (2020).
from aijack.defense import VIB, mib_loss

net = VIB(encoder, decoder, dim_of_latent_space, num_samples=samples_amount)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

for x_batch, y_batch in tqdm(train_loader):
    optimizer.zero_grad()
    y_pred, result_dict = net(x_batch)
    loss = net.loss(y_batch, result_dict)
    loss.backward()
    optimizer.step()
```
