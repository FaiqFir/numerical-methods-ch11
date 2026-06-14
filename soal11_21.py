"""
Soal 11.21 - Augmentasi Matriks [A|I] (Setara perintah MATLAB satu baris)
Sumber: Numerical Methods for Engineers, Chapra & Canale

MATLAB: Aug = [A, eye(size(A))]
Python/NumPy ekuivalen: Aug = np.hstack([A, np.eye(len(A))])
"""
import numpy as np

print("=" * 55)
print("Soal 11.21 - Augmentasi Matriks [A | I]")
print("=" * 55)

# Contoh matriks A
A = np.array([
    [2.0, 1.0, -1.0],
    [-3.0, -1.0, 2.0],
    [-2.0, 1.0, 2.0]
], dtype=float)

n = len(A)
I = np.eye(n)

print("\nMatriks A:")
print(A)
print(f"\nDimensi A: {A.shape}")

# =============================================
# MATLAB ONE-LINER EQUIVALENT:
# Aug = [A, eye(size(A,1))]
# =============================================
Aug = np.hstack([A, I])

print("\n--- Satu baris perintah Python (setara MATLAB) ---")
print("Aug = np.hstack([A, np.eye(len(A))])")
print("\nHasil Aug = [A | I]:")
print(Aug)

print(f"\nDimensi Aug: {Aug.shape}  (dari {n}x{n} menjadi {n}x{2*n})")

# Verifikasi: Gauss-Jordan pada Aug memberikan invers
print("\n--- Bonus: Menghitung invers via Gauss-Jordan pada Aug ---")
M = Aug.copy()

for col in range(n):
    # Pivot
    pivot = M[col][col]
    M[col] /= pivot
    for row in range(n):
        if row != col:
            M[row] -= M[row][col] * M[col]

A_inv_gauss = M[:, n:]
print("Invers A (via Gauss-Jordan):")
print(np.round(A_inv_gauss, 6))

A_inv_np = np.linalg.inv(A)
print("\nInvers A (numpy):")
print(np.round(A_inv_np, 6))

print(f"\nMax error: {np.max(np.abs(A_inv_gauss - A_inv_np)):.2e}")

# Contoh berbagai ukuran
print("\n--- Contoh berbagai ukuran matriks ---")
for size in [2, 3, 4, 5]:
    A_test = np.random.randint(1, 10, (size, size)).astype(float)
    Aug_test = np.hstack([A_test, np.eye(size)])
    print(f"  A ({size}x{size}) → Aug ({size}x{2*size}) ✓")
