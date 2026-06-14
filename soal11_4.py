"""
Soal 11.4 - Verifikasi Cholesky Decomposition dari Example 11.2
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def cholesky_decomposition(A):
    """
    Melakukan Cholesky Decomposition: A = L * L^T
    untuk matriks simetris positif definit.
    """
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

# Matriks simetris dari Example 11.2 (Chapra)
# A = [[6, 15, 55],
#      [15, 55, 225],
#      [55, 225, 979]]
A = np.array([
    [6,  15,  55],
    [15, 55,  225],
    [55, 225, 979]
], dtype=float)

print("=" * 55)
print("Soal 11.4 - Verifikasi Cholesky Decomposition")
print("=" * 55)
print("\nMatriks A:")
print(A)

L = cholesky_decomposition(A)
print("\nMatriks L (Lower Triangular):")
print(np.round(L, 6))

print("\nMatriks L^T (Transpose):")
print(np.round(L.T, 6))

# Verifikasi: L * L^T seharusnya = A
product = L @ L.T
print("\nVerifikasi L * L^T:")
print(np.round(product, 6))

print("\nMatriks A asli:")
print(A)

error = np.max(np.abs(product - A))
print(f"\nMax error |L*L^T - A|: {error:.2e}")

if error < 1e-8:
    print("✓ Verifikasi BERHASIL: L * L^T = A")
else:
    print("✗ Verifikasi GAGAL")

# Bandingkan dengan numpy Cholesky
L_np = np.linalg.cholesky(A)
print("\nCholesky numpy:")
print(np.round(L_np, 6))
print("Max error vs numpy:", np.max(np.abs(L - L_np)))
