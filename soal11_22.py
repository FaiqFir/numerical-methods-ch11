"""
Soal 11.22 - Bentuk Matriks, Solusi, Transpose, dan Invers
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

print("=" * 55)
print("Soal 11.22 - Bentuk Matriks, Solusi, Transpose, Invers")
print("=" * 55)

# Sistem persamaan dari soal 11.22 (tipikal Chapra)
# 4x1 + 2x2 - x3 = -3
# -2x1 + 5x2 - x3 = -7
# x1 - 2x2 + 6x3 = 6

print("\nSistem persamaan:")
print("  4*x1 + 2*x2 - 1*x3 = -3")
print(" -2*x1 + 5*x2 - 1*x3 = -7")
print("  1*x1 - 2*x2 + 6*x3 =  6")

A = np.array([
    [ 4.0,  2.0, -1.0],
    [-2.0,  5.0, -1.0],
    [ 1.0, -2.0,  6.0]
], dtype=float)

b = np.array([-3.0, -7.0, 6.0], dtype=float)

print("\nBentuk Matriks [A]{x} = {b}:")
print("A =")
print(A)
print("b =", b)

# Solusi
x = np.linalg.solve(A, b)
print("\n--- Solusi ---")
print(f"x1 = {x[0]:.6f}")
print(f"x2 = {x[1]:.6f}")
print(f"x3 = {x[2]:.6f}")

# Verifikasi
print("\nVerifikasi A*x = b:")
print("A*x =", np.round(A @ x, 6))
print("b   =", b)

# Transpose
A_T = A.T
print("\n--- Transpose A^T ---")
print(A_T)

# Properti transpose
print("\nProperti Transpose:")
print(f"  (A^T)^T = A? {np.allclose((A_T).T, A)}")
print(f"  (A^T)[i,j] = A[j,i]? {np.allclose(A_T[1,2], A[2,1])}")

# Invers
A_inv = np.linalg.inv(A)
print("\n--- Invers A^(-1) ---")
print(np.round(A_inv, 6))

# Verifikasi invers
print("\nVerifikasi A * A^(-1) = I:")
product = A @ A_inv
print(np.round(product, 8))

print("\nVerifikasi A^(-1) * A = I:")
print(np.round(A_inv @ A, 8))

# Determinan
det = np.linalg.det(A)
print(f"\nDeterminan A = {det:.6f}")
print(f"Matriks invertible? {'Ya' if abs(det) > 1e-10 else 'Tidak (singular)'}")

# Eigenvalues
eigenvalues = np.linalg.eigvals(A)
print(f"\nEigenvalues: {np.round(eigenvalues, 4)}")
