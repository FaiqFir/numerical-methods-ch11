"""
Soal 11.7 - Cholesky Decomposition + Analisis Eqs. (11.3) dan (11.4)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Persamaan (11.3): l_jj = sqrt(a_jj - sum(l_jk^2, k=1..j-1))
Persamaan (11.4): l_ij = (a_ij - sum(l_ik*l_jk, k=1..j-1)) / l_jj  untuk i > j
"""
import numpy as np

def cholesky_with_analysis(A):
    """Cholesky dengan verifikasi persamaan (11.3) dan (11.4)."""
    n = len(A)
    L = np.zeros((n, n))

    for j in range(n):
        # Eq. 11.3: diagonal
        s = sum(L[j][k]**2 for k in range(j))
        L[j][j] = np.sqrt(A[j][j] - s)

        # Eq. 11.4: off-diagonal
        for i in range(j+1, n):
            s2 = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / L[j][j]

    return L

# Matriks 2x2 sederhana dari soal 11.7
A = np.array([
    [4.0, 2.0],
    [2.0, 3.0]
], dtype=float)

print("=" * 55)
print("Soal 11.7 - Cholesky + Verifikasi Eqs. (11.3) & (11.4)")
print("=" * 55)
print("\nMatriks A:")
print(A)
print(f"\nUkuran: {A.shape[0]}x{A.shape[1]}")

# Cek apakah simetris
print(f"\nApakah simetris: {np.allclose(A, A.T)}")

# Cek positif definit (eigenvalues > 0)
eigenvalues = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Apakah positif definit: {all(eigenvalues > 0)}")

L = cholesky_with_analysis(A)
print("\nMatriks L:")
print(np.round(L, 6))

print("\n--- Verifikasi Persamaan (11.3) ---")
print(f"  l_11 = sqrt(a_11) = sqrt({A[0][0]}) = {np.sqrt(A[0][0]):.6f}")
print(f"  l_22 = sqrt(a_22 - l_21^2) = sqrt({A[1][1]} - {L[1][0]**2:.4f}) = {L[1][1]:.6f}")

print("\n--- Verifikasi Persamaan (11.4) ---")
print(f"  l_21 = a_21 / l_11 = {A[1][0]} / {L[0][0]:.6f} = {L[1][0]:.6f}")

# Rekonstruksi
product = L @ L.T
print("\nRekonstruksi L * L^T:")
print(np.round(product, 6))
error = np.max(np.abs(product - A))
print(f"Max error: {error:.2e}")
if error < 1e-10:
    print("✓ Hasil masuk akal sesuai Eqs. (11.3) dan (11.4)")
