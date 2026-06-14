"""
Soal 11.1 - Thomas Algorithm for Tridiagonal System
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system using the Thomas algorithm.
    a = lower diagonal (len n-1)
    b = main diagonal (len n)
    c = upper diagonal (len n-1)
    d = right-hand side (len n)
    """
    n = len(d)
    # Copy arrays to avoid modifying originals
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)

    # Forward sweep
    for i in range(1, n):
        factor = a[i-1] / b[i-1]
        b[i] -= factor * c[i-1]
        d[i] -= factor * d[i-1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x

# (a) Tridiagonal system from Example 11.1 context
# System:
# 2.04*x1 - 1.0*x2 = 40.8
# -1.0*x1 + 2.04*x2 - 1.0*x3 = 0.8
# -1.0*x2 + 2.04*x3 - 1.0*x4 = 0.8
# -1.0*x3 + 2.04*x4 = 200.8
print("=" * 50)
print("Soal 11.1(a) - Sistem Tridiagonal (Thomas Algorithm)")
print("=" * 50)

# Main diagonal
b = [2.04, 2.04, 2.04, 2.04]
# Upper diagonal
c = [-1.0, -1.0, -1.0]
# Lower diagonal
a = [-1.0, -1.0, -1.0]
# RHS
d = [40.8, 0.8, 0.8, 200.8]

x = thomas_algorithm(a, b, c, d)
print("Solusi x:", x)

# Verifikasi dengan numpy
A = np.array([
    [2.04, -1.0,  0.0,  0.0],
    [-1.0,  2.04, -1.0,  0.0],
    [0.0,  -1.0,  2.04, -1.0],
    [0.0,   0.0, -1.0,  2.04]
])
d_vec = np.array([40.8, 0.8, 0.8, 200.8])
x_np = np.linalg.solve(A, d_vec)
print("Verifikasi numpy:", x_np)
print("Max error:", np.max(np.abs(x - x_np)))

# (b) Example 11.3 - Gauss-Seidel tridiagonal
print("\n" + "=" * 50)
print("Soal 11.1(b) - Tridiagonal dengan Gauss-Seidel")
print("=" * 50)

def gauss_seidel_tridiagonal(A, b_rhs, tol=1e-6, max_iter=100):
    n = len(b_rhs)
    x = np.zeros(n)
    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b_rhs[i] - sigma) / A[i][i]
        # Check convergence
        if np.max(np.abs(x - x_old)) / (np.max(np.abs(x)) + 1e-15) * 100 < tol:
            print(f"  Konvergen pada iterasi {iteration+1}")
            break
    return x

x_gs = gauss_seidel_tridiagonal(A, d_vec, tol=5.0)
print("Solusi Gauss-Seidel:", x_gs)
