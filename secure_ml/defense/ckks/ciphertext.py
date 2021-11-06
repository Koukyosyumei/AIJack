from numpy.polynomial import Polynomial

from .plaintext import CKKSPlaintext
from .utils import polydiv_coef


class CKKSCiphertext:
    def __init__(self, c0: Polynomial, c1: Polynomial, N: int, q: int, scale: int):
        self.c0 = c0
        self.c1 = c1
        self.N = N
        self.q = q
        self.scale = scale

        self.poly_modulo = Polynomial(
            [1 if i == 0 or i == N else 0 for i in range(N + 1)]
        )

    def __add__(self, other):
        if type(other) == CKKSPlaintext:
            return CKKSCiphertext(
                polydiv_coef(self.c0 + other.p, self.q) % self.poly_modulo,
                polydiv_coef(self.c1 + other.p, self.q) % self.poly_modulo,
                self.N,
                self.q,
                self.scale,
            )
        elif type(other) == CKKSCiphertext:
            return CKKSCiphertext(
                polydiv_coef(self.c0 + other.c0, self.q) % self.poly_modulo,
                polydiv_coef(self.c1 + other.c1, self.q) % self.poly_modulo,
                self.N,
                self.q,
                self.scale,
            )
        else:
            raise TypeError("error!")

    def __mul__(self, other):
        if type(other) == CKKSPlaintext:
            return CKKSCiphertext(
                polydiv_coef((self.c0 * other.p) / self.scale, self.q)
                % self.poly_modulo,
                polydiv_coef((self.c1 * other.p) / self.scale, self.q)
                % self.poly_modulo,
                self.N,
                self.q,
                self.scale,
            )
        else:
            raise TypeError("error!")
