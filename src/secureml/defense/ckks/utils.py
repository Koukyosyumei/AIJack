import numpy as np
from numpy.polynomial import Polynomial


def round_coordinates(coordinates):
    """Gives the integral rest."""
    coordinates = coordinates - np.floor(coordinates)
    return coordinates


def coordinate_wise_random_rounding(coordinates):
    """Rounds coordinates randonmly."""
    r = round_coordinates(coordinates)
    f = np.array([np.random.choice([c, c - 1], 1, p=[1 - c, c]) for c in r]).reshape(-1)

    rounded_coordinates = coordinates - f
    rounded_coordinates = [int(coeff) for coeff in rounded_coordinates]
    return rounded_coordinates


def polydiv_coef(p: Polynomial, mod: int) -> Polynomial:
    return Polynomial([c % mod for c in p.coef])
