import numpy as np
from numpy.polynomial import Polynomial

from .plaintext import CKKSPlaintext
from .utils import coordinate_wise_random_rounding


class CKKSEncoder:
    """Basic CKKS encoder to encode complex vectors into polynomials.
    This class is mainly based on https://blog.openmined.org/ckks-explained-part-1-simple-encoding-and-decoding/
    """

    def __init__(self, M: int, scale: float):
        """Initialization of the encoder for M a power of 2.

        Args:
            M:
            scale:

        Attributed:
            xi: M-th root of unity used as a basis
            N:
            A:
            sigma_R_basis
        """
        self.M = M
        self.scale = scale

        self.xi = np.exp(2 * np.pi * 1j / M)
        self.N = M // 2
        self.A = self._vandermonde()
        self.sigma_R_basis = self.A.T

    def _vandermonde(self) -> np.array:
        """Get vandermonde matrix of a m-th root of unity.

        Returns:
            np.array: a vandermonde matrix (N^N)
        """
        roots = np.array([self.xi ** (2 * i + 1) for i in range(self.N)])
        matrix = np.array([roots**j for j in range(self.N)])
        return matrix.T

    def sigma_inverse(self, z: np.array) -> Polynomial:
        """Encodes the vector z (C^{N/2}) to a polinomial ().

        Args:
            z (np.array): target vector

        Returns:
            Polynomial: the encoded polinoimial
        """
        return Polynomial(np.linalg.solve(self.A, z))

    def sigma(self, p: Polynomial) -> np.array:
        """Decodes a polynomial to a vector (C^{N/2})

        Args:
            p: encoded polinoimial

        Returns:
            np.array: decoded vector
        """
        return np.array([p(self.xi ** (2 * i + 1)) for i in range(self.N)])

    def pi(self, z: np.array) -> np.array:
        """Projects a vector of H into C^{N/2}.

        Args:
            z:

        Returns:
            np.array:
        """
        return z[: self.N // 2]

    def pi_inverse(self, z: np.array) -> np.array:
        """Expands a vector of C^{N/2} by expanding it with its
        complex conjugate."""

        z_conjugate = z[::-1]
        z_conjugate = [np.conjugate(x) for x in z_conjugate]
        return np.concatenate([z, z_conjugate])

    def compute_basis_coordinates(self, z):
        """Computes the coordinates of a vector with respect to the orthogonal lattice basis."""
        output = np.array(
            [np.real(np.vdot(z, b) / np.vdot(b, b)) for b in self.sigma_R_basis]
        )
        return output

    def sigma_R_discretization(self, z):
        """Projects a vector on the lattice using coordinate wise random rounding."""
        coordinates = self.compute_basis_coordinates(z)
        rounded_coordinates = coordinate_wise_random_rounding(coordinates)
        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates)
        return y

    def encode(self, z: np.array) -> CKKSPlaintext:
        """Encodes a vector by expanding it first to H,
        scale it, project it on the lattice of sigma(R), and performs
        sigma inverse.
        """
        pi_z = self.pi_inverse(z)
        scaled_pi_z = self.scale * pi_z
        rounded_scale_pi_zi = self.sigma_R_discretization(scaled_pi_z)
        p = self.sigma_inverse(rounded_scale_pi_zi)

        # We round it afterwards due to numerical imprecision
        coef = np.round(np.real(p.coef)).astype(int)
        pt = CKKSPlaintext(Polynomial(coef), self.N, self.scale)
        return pt

    def decode(self, pt: CKKSPlaintext) -> np.array:
        """Decodes a polynomial by removing the scale,
        evaluating on the roots, and project it on C^(N/2)"""
        rescaled_p = pt.p / self.scale
        z = self.sigma(rescaled_p)
        pi_z = self.pi(z)
        return pi_z
