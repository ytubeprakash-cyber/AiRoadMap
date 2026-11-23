import numpy as np
from modelcraft_lab.linalg import dot, matmul


def test_dot_matches_numpy():
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    assert dot(a, b) == np.dot(a, b) == 32.0


def test_matmul_matches_numpy_and_shape():
    rng = np.random.default_rng(seed=0)
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(3, 4))
    C = np.array(matmul(A.tolist(), B.tolist()))
    assert C.shape == (2, 4)
    assert np.allclose(C, A @ B)


def test_batched_shapes():
    x = np.zeros((8, 16, 32))
    W = np.zeros((32, 64))
    assert (x @ W).shape == (8, 16, 64)