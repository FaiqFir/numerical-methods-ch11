"""
Soal 11.5 - Cholesky Decomposition untuk Sistem Simetris
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def cholesky_decomposition(A):
    """Cholesky: A = L * L^T"""
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val < 0:
                    raise ValueError("Matriks tidak positif definit!")
                L[i][j] = np.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def forward_substitution(L, b):
    """Solves L*y = b"""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def back_substitution(U, y):
    """Solves U*x = y"""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

# Sistem simetris dari soal 11.5
# Tipikal matriks dari konteks buku Chapra soal 11.5
A = np.array([
    [2.0, -1.0,  0.0],
    [-1.0,  2.0, -1.0],
    [0.0, -1.0,  2.0]
], dtype=float)

b = np.array([1.0, 0.0, 1.0], dtype=float)

print("=" * 55)
print("Soal 11.5 - Cholesky Decomposition + Solve")
print("=" * 55)
print("\nMatriks A (simetris):")
print(A)
print("\nVector b:", b)

# Cholesky decomposition
L = cholesky_decomposition(A)
print("\nMatriks L:")
print(np.round(L, 6))
print("\nMatriks L^T:")
print(np.round(L.T, 6))

# Solve: A*x = b  -->  L*(L^T*x) = b
# Step 1: L*y = b
y = forward_substitution(L, b)
print("\nVector y (dari L*y = b):", np.round(y, 6))

# Step 2: L^T*x = y
x = back_substitution(L.T, y)
print("Solusi x:", np.round(x, 6))

# Verifikasi
residual = A @ x - b
print("\nVerifikasi A*x - b:", np.round(residual, 10))
print("Max residual:", np.max(np.abs(residual)))

# Bandingkan numpy
x_np = np.linalg.solve(A, b)
print("Solusi numpy:", x_np)
print("Max error:", np.max(np.abs(x - x_np)))
