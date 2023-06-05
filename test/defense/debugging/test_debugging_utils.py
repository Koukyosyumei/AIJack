def test_hungarian():
    import numpy as np

    from aijack.defense.debugging.utils import Hungarian

    a = [5, 4, 7, 6]
    b = [6, 7, 3, 2]
    c = [8, 11, 2, 5]
    d = [9, 8, 6, 7]
    mat = np.array([a, b, c, d])
    ground_truth_assignment = [(0, 0), (1, 3), (2, 2), (3, 1)]

    h = Hungarian()
    assert ground_truth_assignment == h.compute(mat)
