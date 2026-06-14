"""
Soal 11.11 - Gauss-Seidel sampai percent relative error < 5%
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def gauss_seidel(A, b, tol=5.0, max_iter=200, verbose=True):
    """Gauss-Seidel dengan output iterasi lengkap."""
    n = len(b)
    x = np.zeros(n)

    if verbose:
        header = f"{'Iter':>5} | " + " ".join([f"{'x'+str(i+1):>12}" for i in range(n)]) + f" | {'Max %err':>10}"
        print(header)
        print("-" * len(header))

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-15:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100

        max_ea = np.max(ea)

        if verbose:
            print(f"{iteration+1:>5} | " + " ".join([f"{xi:>12.6f}" for xi in x]) + f" | {max_ea:>10.4f}%")

        if max_ea < tol:
            if verbose:
                print(f"\n✓ Konvergen pada iterasi {iteration+1} (max error = {max_ea:.4f}% < {tol}%)")
            return x, iteration + 1

    if verbose:
        print(f"\n⚠ Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter

# Sistem dari soal 11.11
# 10*x1 - 2*x2 - x3 = 27
# -2*x1 + 10*x2 - x3 = -61.5
# -x1 - x2 + 5*x3 = -21.5
A = np.array([
    [10.0, -2.0, -1.0],
    [-2.0,  10.0, -1.0],
    [-1.0,  -1.0,  5.0]
], dtype=float)

b = np.array([27.0, -61.5, -21.5], dtype=float)

print("=" * 60)
print("Soal 11.11 - Gauss-Seidel (es = 5%)")
print("=" * 60)
print("\nSistem persamaan:")
labels = ['x1', 'x2', 'x3']
for i in range(3):
    terms = []
    for j in range(3):
        if A[i][j] != 0:
            if j == 0:
                terms.append(f"{A[i][j]}*x{j+1}")
            elif A[i][j] > 0:
                terms.append(f"+ {A[i][j]}*x{j+1}")
            else:
                terms.append(f"- {abs(A[i][j])}*x{j+1}")
    print(f"  {' '.join(terms)} = {b[i]}")

# Cek dominansi diagonal
print("\nCek dominansi diagonal:")
for i in range(3):
    diag = abs(A[i][i])
    off = sum(abs(A[i][j]) for j in range(3) if j != i)
    status = "✓ dominan" if diag > off else "✗ tidak dominan"
    print(f"  Baris {i+1}: |{diag}| > |{off}|? {status}")

print("\n--- Iterasi Gauss-Seidel ---")
x, n_iter = gauss_seidel(A, b, tol=5.0)

x_exact = np.linalg.solve(A, b)
print(f"\nSolusi akhir: x1={x[0]:.4f}, x2={x[1]:.4f}, x3={x[2]:.4f}")
print(f"Solusi eksak: x1={x_exact[0]:.4f}, x2={x_exact[1]:.4f}, x3={x_exact[2]:.4f}")
