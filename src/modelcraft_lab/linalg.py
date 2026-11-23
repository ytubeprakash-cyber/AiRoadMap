def dot(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    m, k = len(A), len(A[0])
    k2, n = len(B), len(B[0])
    assert k == k2, f"inner dims must match: {k} vs {k2}"
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = dot(A[i], [B[r][j] for r in range(k)])
    return C