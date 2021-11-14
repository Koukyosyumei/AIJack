from numpy.polynomial import Polynomial

from .ciphertext import CKKSCiphertext
from .plaintext import CKKSPlaintext
from .random import gen_binary_poly, gen_normal_poly, gen_uniform_poly
from .utils import polydiv_coef


class CKKSEncrypter:
    def __init__(self, encoder, q):
        self.N = encoder.N
        self.q = q
        self.encoder = encoder
        self.scale = encoder.scale

        self.poly_mod = Polynomial(
            [1 if i == 0 or i == self.N else 0 for i in range(self.N + 1)]
        )

        self.pk = None
        self.sk = None

    def set_pk(self, pk):
        self.pk = pk

    def set_sk(self, sk):
        self.sk = sk

    def keygen(self, size):
        """Generate a public and secret keys
        Args:
            size: size of the polynoms for the public and secret keys.
            q: coefficient modulus.
            poly_mod: polynomial modulus.
        Returns:
            Public and secret key.
        """
        self.sk = gen_binary_poly(size)
        a = gen_uniform_poly(size, self.q)
        e = gen_normal_poly(size, std=1)
        b = (
            polydiv_coef(polydiv_coef(-a * self.sk, self.q) % self.poly_mod - e, self.q)
            % self.poly_mod
        )
        self.pk = (b, a)
        return self.pk, self.sk

    def encrypt_from_plaintext(self, pt):
        """Encrypt an integer.
        Args:
            pk: public-key.
            pt: plaintext to be encrypted.
        Returns:
            Tuple representing a ciphertext.
        """
        return CKKSCiphertext(
            polydiv_coef(pt.p + self.pk[0], self.q) % self.poly_mod,
            self.pk[1],
            self.N,
            self.q,
            self.scale,
        )

    def decrypt_to_plaintext(self, ct):
        return CKKSPlaintext(
            polydiv_coef(
                ct.c0 + polydiv_coef(ct.c1 * self.sk, self.q) % self.poly_mod, self.q
            )
            % self.poly_mod,
            self.N,
            self.scale,
        )

    def encrypt(self, z):
        return self.encrypt_from_plaintext(self.encoder.encode(z))

    def decrypt(self, ct):
        return self.encoder.decode(self.decrypt_to_plaintext(ct))
