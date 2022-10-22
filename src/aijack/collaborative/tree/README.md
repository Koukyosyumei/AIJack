## Federated Tree-based Learning


### Install

```
cd tree
pip install -e .
```

### Import

```Python
from aijack_secureboost import (  # noqa: F401
    XGBoostParty,
    XGBoostClassifier,
    SecureBoostParty,
    SecureBoostClassifier,
    PaillierKeyGenerator
)
```

### XGBoost

```Python

min_leaf = 1
depth = 3
learning_rate = 0.4
boosting_rounds = 2
lam = 1.0
gamma = 0.0
eps = 1.0
min_child_weight = -1 * float("inf")
subsample_cols = 1.0

clf = XGBoostClassifier(2,subsample_cols,min_child_weight,depth,min_leaf,
                  learning_rate,boosting_rounds,lam,gamma,eps,-1,0,1.0,1,True)

x1 = [12, 32, 15, 24, 20, 25, 17, 16]
x1 = [[x] for x in x1]
x2 = [1, 1, 0, 0, 1, 1, 0, 1]
x2 = [[x] for x in x2]
y = [1, 0, 1, 0, 1, 1, 0, 1]

p1 = XGBoostParty(x1, 2, [0], 0, min_leaf, subsample_cols, 256, False, 0)
p2 = XGBoostParty(x2, 2, [1], 1, min_leaf, subsample_cols, 256, False, 0)

parties = [p1, p2]

clf.fit(parties, y)

X_new = [[12, 1], [32, 1], [15, 0], [24, 0], [20, 1], [25, 1], [17, 0], [16, 1]]

clf.predict_proba(X)
```

### SecureBoost

```Python

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
