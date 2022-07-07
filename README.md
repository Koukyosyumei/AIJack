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

# Quick Start

This package implements algorithms for AI security such as Model Inversion, Poisoning Attack, Evasion Attack, Differential Privacy, and Homomorphic Encryption.

## Install

```
# pip install pybind11 (uncomment if necessary)
pip install git+https://github.com/Koukyosyumei/AIJack
```

## Supported Algorithms

### 1. Collaborative Learning

>>:globe_with_meridians: Train a single model without sharing the private datasets of multiple clients.

#### 1.1. NN

- FedAVG ([example](example/model_inversion/soteria.py)) ([paper](https://arxiv.org/abs/1602.05629))
- FedProx ([paper](https://arxiv.org/abs/1812.06127))
- FedKD ([example](test/collaborative/fedkd/test_fedkd.py)) ([paper](https://arxiv.org/abs/2108.13323))
- FedMD ([paper](https://arxiv.org/abs/1910.03581))
- FedGEMS ([paper](https://arxiv.org/abs/2110.11027))
- DSFL ([paper](https://arxiv.org/abs/2008.06180))
- SplitNN ([example](example/label_leakage/label_leakage.py)) ([paper](https://arxiv.org/abs/1812.00564))

#### 1.2. Tree

- [WIP] SecureBoost ([example](test/collaborative/secureboost/test_secureboost.py)) ([paper](https://arxiv.org/abs/1901.08755))

### 2. Attack

#### 2.1. Model Inversion Attack

>>:zap: Reconstruct the private training dataset from the victim's model.

- MI-FACE ([example](example/model_inversion/mi_face.py)) ([paper](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677))
- DLG ([example](example/model_inversion/gradient_inversion_attack.md)) ([paper](https://papers.nips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html))
- iDLG ([example](example/model_inversion/gradient_inversion_attack.md)) ([paper](https://arxiv.org/abs/2001.02610))
- GS ([example](example/model_inversion/gradient_inversion_attack.md)) ([paper](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html))
- CPL ([example](example/model_inversion/gradient_inversion_attack.md)) ([paper](https://arxiv.org/abs/2004.10397))
- GradInversion ([example](example/model_inversion/gradient_inversion_attack.md)) ([paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_See_Through_Gradients_Image_Batch_Recovery_via_GradInversion_CVPR_2021_paper.pdf))
- GAN attack ([example](example/model_inversion/gan_attack.py)) ([paper](https://arxiv.org/abs/1702.07464))

#### 2.2. Membership Inference Attack

>>:zap: Determine whether the model’s training dataset contains the target record.

- Blak-box attack with shadow models ([example](example/membership_inference/membership_inference_CIFAR10.ipynb)) ([paper](https://arxiv.org/abs/1610.05820))

#### 2.3. Label Leakage Attack

>>:zap: Infer the label information of the dataset.

- Norm attack ([example](example/label_leakage/label_leakage.py)) ([paper](https://arxiv.org/abs/2102.08504))

#### 2.4. Evasion Attack

>>:zap: Generate data that the victim model cannot classify correctly.

- Gradient descent attacks ([example](example/adversarial_example/example_evasion_attack_svm.ipynb)) ([paper](https://arxiv.org/abs/1708.06131))

#### 2.5. Poisoning Attack

>>:zap: Inject malicious data into the training dataset to control the behavior of the trained models.

- Poisoning attack against support vector machines ([example](example/adversarial_example/example_poison_attack.ipynb)) ([paper](https://arxiv.org/abs/1206.6389))

### 3. Defense

#### 3.1. Differential Privacy

>>:closed_lock_with_key: Provide statistical privacy guarantee.

- DPSGD ([example](example/model_inversion/mi_face_differential_privacy.py)) ([paper](https://arxiv.org/abs/1607.00133))

#### 3.2 Homomorphic Encryption

>>:closed_lock_with_key: Perform mathematical operations on encrypted data

- Paillier ([README](src/aijack/defense/paillier/README.md))
- [WIP] CKKS ([example](test/defense/ckks/test_core.py))

#### 3.3. Others

- Soteria ([example](example/model_inversion/soteria.py)) ([paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.pdf))
- MID ([example](example/model_inversion/mid.ipynb)) ([paper](https://arxiv.org/abs/2009.05241))

## Resources

[WIP] [Official documentations (https://koukyosyumei.github.io/AIJack)](https://koukyosyumei.github.io/AIJack) \
[WIP] [Examples](https://github.com/Koukyosyumei/AIJack/tree/main/example)

## Contact

welcome2aijack[@]gmail.com

-----------------------------------------------------------------------

## Examples of Usage

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

- MID

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

- Paillier

Key generation

```Python
>>> from aijack_paillier import PaillierKeyGenerator
>>> keygenerator = PaillierKeyGenerator(512)
>>> pk, sk = keygenerator.generate_keypair()

>>> pk.get_publickeyvalues()
('88227558341633487968602031548884474473159932637542421321125729261006034700025348204644005240122191376832590701931729414754607546418236854601939515234119196121342390460128908962474840402676824539316895354970651799849034208946592214675397708295317235925162498035034951208255290311663747050727479504325436699623', '88227558341633487968602031548884474473159932637542421321125729261006034700025348204644005240122191376832590701931729414754607546418236854601939515234119196121342390460128908962474840402676824539316895354970651799849034208946592214675397708295317235925162498035034951208255290311663747050727479504325436699624')
```

Encrypt & Decrypt

```Python
>>> ct_1 = pk.encrypt(13)
>>> pk.decrypt4int(ct_1)
13
```

Arithmetic operation

```Python
>>> ct_2 = ct_1 * 2
>>> pk.decrypt4int(ct_2)
26

>>> ct_3 = ct_1 + 5.6
>>> sk.decrypt2float(ct_3)
18.6

>>> ct_4 = pk.encrypt(18)
>>> ct_5 = ct_1 + ct_4
>>> sk.decrypt2int(ct_5)
31
```
