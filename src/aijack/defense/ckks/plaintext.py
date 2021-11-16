from numpy.polynomial import Polynomial


class CKKSPlaintext:
    def __init__(self, p: Polynomial, N: int, scale: int):
        self.p = p
        self.N = N
        self.scale = scale

        self.poly_modulo = Polynomial(
            [1 if i == 0 or i == N else 0 for i in range(N + 1)]
        )

    def __add__(self, other):
        if type(other) in [int, float]:
            return CKKSPlaintext(self.p + other, self.N, self.scale)
        elif type(other) == CKKSPlaintext:
            return CKKSPlaintext(self.p + other.p, self.N, self.scale)
        else:
            raise TypeError("error!")

    def __mul__(self, other):
        if type(other) in [int, float]:
            return CKKSPlaintext(self.p * other, self.N, self.scale)
        elif type(other) == CKKSPlaintext:
            return CKKSPlaintext(
                ((self.p * other.p) / self.scale) % self.poly_modulo, self.N, self.scale
            )
        else:
            raise TypeError("error!")
