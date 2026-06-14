"""
Soal 11.3 - Crank-Nicolson Tridiagonal System (Thomas Algorithm)
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Thomas Algorithm untuk sistem tridiagonal.
    a = lower diagonal (len n-1)
    b = main diagonal (len n)
    c = upper diagonal (len n-1)
    d = RHS vector (len n)
    """
    n = len(d)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # Forward sweep
    for i in range(1, n):
        m = a[i-1] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x

print("=" * 55)
print("Soal 11.3 - Crank-Nicolson Tridiagonal (Thomas Algorithm)")
print("=" * 55)

# Crank-Nicolson tridiagonal system (typical form)
# For a 2nd-order PDE finite difference: 
# -r*T_{i-1} + (1+2r)*T_i - r*T_{i+1} = r*T_{i-1}^old + (1-2r)*T_i^old + r*T_{i+1}^old
# 
# Contoh umum sistem Crank-Nicolson 5 persamaan:
# Diagonal utama = 2.04, upper/lower diagonal = -1.0
# (nilai-nilai tipikal dari buku teks)

# Sistem tridiagonal dari konteks soal 11.3 di buku Chapra:
# Matriks 5x5, koefisien tipikal Crank-Nicolson
n = 5
b_diag = [2.0393, 2.0393, 2.0393, 2.0393, 2.0393]  # main diagonal
a_diag = [-1.0, -1.0, -1.0, -1.0]                   # lower diagonal
c_diag = [-1.0, -1.0, -1.0, -1.0]                   # upper diagonal
d_rhs  = [40.8, 0.8, 0.8, 0.8, 200.8]               # RHS

print("\nSistem Tridiagonal:")
print(f"  Main diagonal (b): {b_diag}")
print(f"  Lower diagonal (a): {a_diag}")
print(f"  Upper diagonal (c): {c_diag}")
print(f"  RHS (d): {d_rhs}")

x = thomas_algorithm(a_diag, b_diag, c_diag, d_rhs)
print("\nSolusi dengan Thomas Algorithm:")
for i, xi in enumerate(x):
    print(f"  T_{i+1} = {xi:.6f}")

# Verifikasi dengan numpy
A_full = np.zeros((n, n))
for i in range(n):
    A_full[i][i] = b_diag[i]
for i in range(n-1):
    A_full[i+1][i] = a_diag[i]
    A_full[i][i+1] = c_diag[i]

x_np = np.linalg.solve(A_full, d_rhs)
print("\nVerifikasi numpy:", x_np)
print("Max error:", np.max(np.abs(x - x_np)))
