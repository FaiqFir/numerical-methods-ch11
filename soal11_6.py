"""
Soal 11.6 - Cholesky Decomposition Manual (Symmetric System)
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def cholesky_manual(A):
    """
    Cholesky decomposition dengan menampilkan langkah-langkah.
    A = L * L^T
    """
    n = len(A)
    L = np.zeros((n, n))

    print("--- Langkah-langkah Cholesky Decomposition ---")
    for j in range(n):
        # Diagonal element
        s = sum(L[j][k]**2 for k in range(j))
        L[j][j] = np.sqrt(A[j][j] - s)
        print(f"  L[{j+1}][{j+1}] = sqrt({A[j][j]:.4f} - {s:.4f}) = {L[j][j]:.6f}")

        # Off-diagonal elements
        for i in range(j + 1, n):
            s2 = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / L[j][j]
            print(f"  L[{i+1}][{j+1}] = ({A[i][j]:.4f} - {s2:.4f}) / {L[j][j]:.6f} = {L[i][j]:.6f}")

    return L

# Sistem simetris dari soal 11.6
# Matriks 3x3 simetris tipikal dari buku
A = np.array([
    [1.0,  1.0,  1.0],
    [1.0,  5.0,  5.0],
    [1.0,  5.0, 14.0]
], dtype=float)

print("=" * 55)
print("Soal 11.6 - Cholesky Decomposition Manual")
print("=" * 55)
print("\nMatriks A:")
print(A)
print()

L = cholesky_manual(A)

print("\nMatriks L:")
print(np.round(L, 6))

print("\nMatriks L^T:")
print(np.round(L.T, 6))

# Verifikasi
product = L @ L.T
print("\nVerifikasi L * L^T:")
print(np.round(product, 6))

error = np.max(np.abs(product - A))
print(f"\nMax error: {error:.2e}")
if error < 1e-10:
    print("✓ Dekomposisi benar!")
