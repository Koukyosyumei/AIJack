# Paillier Encryption Scheme

## Install

You need install `boost` in advance.

```
cd paillier
pip install -e .
```

## Usage

Key generation

```Python
from aijack_paillier import PaillierKeyGenerator
keygenerator = PaillierKeyGenerator(512)
```

Encrypt & Decrypt

```Python
ct_1 = pk.encrypt(13)
pk.decrypt4int(ct_1)
>>> 13
```

Arithmetic operation

```Python
ct_2 = ct_1 * 2
pk.decrypt4int(ct_2)
>>> 26

ct_3 = ct_1 + 5.6
sk.decrypt2float(ct_3)
>>> 18.6

ct_4 = pk.encrypt(18)
ct_5 = ct_1 + ct_4
sk.decrypt2int(ct_5)
>>> 31
```
