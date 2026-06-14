"""
Soal 11.2 - Matrix Inverse via LU Decomposition and Unit Vectors
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np
from scipy.linalg import lu

def lu_decomposition(A):
    """LU Decomposition tanpa pivoting (Doolittle method)."""
    n = len(A)
    L = np.eye(n)
    U = A.copy().astype(float)

    for k in range(n - 1):
        for i in range(k + 1, n):
            if abs(U[k][k]) < 1e-15:
                raise ValueError("Zero pivot encountered!")
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
    return L, U

def forward_substitution(L, b):
    """Solves L*y = b (lower triangular)."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def back_substitution(U, y):
    """Solves U*x = y (upper triangular)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def matrix_inverse_lu(A):
    """Computes matrix inverse using LU decomposition + unit vectors."""
    n = len(A)
    L, U = lu_decomposition(A)
    A_inv = np.zeros((n, n))

    for col in range(n):
        # Unit vector e_i
        e = np.zeros(n)
        e[col] = 1.0

        # Forward: L*y = e
        y = forward_substitution(L, e)
        # Backward: U*x = y
        x = back_substitution(U, y)
        A_inv[:, col] = x

    return A_inv

# Matrix dari Example 11.1
A = np.array([
    [2.04, -1.0,  0.0,  0.0],
    [-1.0,  2.04, -1.0,  0.0],
    [0.0,  -1.0,  2.04, -1.0],
    [0.0,   0.0, -1.0,  2.04]
], dtype=float)

print("=" * 55)
print("Soal 11.2 - Invers Matriks via LU Decomposition")
print("=" * 55)
print("\nMatriks A:")
print(A)

A_inv = matrix_inverse_lu(A)
print("\nInvers A (LU method):")
print(np.round(A_inv, 6))

# Verifikasi: A * A_inv = I
product = A @ A_inv
print("\nVerifikasi A * A_inv (seharusnya mendekati I):")
print(np.round(product, 6))

# Bandingkan dengan numpy
A_inv_np = np.linalg.inv(A)
print("\nInvers A (numpy):")
print(np.round(A_inv_np, 6))
print("\nMax error:", np.max(np.abs(A_inv - A_inv_np)))
