"""
Soal 11.9 - Gauss-Seidel untuk Sistem Reaktor Tersambung (dari Prob. 10.8)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Sistem persamaan untuk konsentrasi dalam seri reaktor:
(c dalam g/m³, RHS dalam g/d)
"""
import numpy as np

def gauss_seidel(A, b, tol=5.0, max_iter=200):
    """
    Gauss-Seidel iterasi dengan percent relative error criterion.
    """
    n = len(b)
    x = np.zeros(n)

    print(f"{'Iter':>5} | " + " ".join([f"{'c'+str(i+1):>12}" for i in range(n)]) + f" | {'Max %err':>10}")
    print("-" * (5 + 14*n + 15))

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        # Percent relative error
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-15:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100

        max_ea = np.max(ea)
        print(f"{iteration+1:>5} | " + " ".join([f"{xi:>12.4f}" for xi in x]) + f" | {max_ea:>10.4f}%")

        if max_ea < tol:
            print(f"\n✓ Konvergen pada iterasi {iteration+1}")
            break

    return x

# Sistem dari Prob. 10.8 (Chapra)
# Reaktor 3 tersambung, koefisien dari buku
# 15c1 - 3c2 - c3 = 3800
# -3c1 + 18c2 - 6c3 = 1200
# -4c1 - c2 + 12c3 = 2350
A = np.array([
    [15.0, -3.0, -1.0],
    [-3.0,  18.0, -6.0],
    [-4.0,  -1.0, 12.0]
], dtype=float)

b = np.array([3800.0, 1200.0, 2350.0], dtype=float)

print("=" * 60)
print("Soal 11.9 - Gauss-Seidel: Reaktor Tersambung")
print("=" * 60)
print("\nSistem persamaan:")
for i in range(len(b)):
    terms = " + ".join([f"{A[i][j]}*c{j+1}" for j in range(len(b)) if A[i][j] != 0])
    print(f"  {terms} = {b[i]}")

# Cek dominansi diagonal
print("\nCek dominansi diagonal:")
for i in range(len(b)):
    diag = abs(A[i][i])
    off_sum = sum(abs(A[i][j]) for j in range(len(b)) if j != i)
    status = "✓" if diag >= off_sum else "✗"
    print(f"  Baris {i+1}: |{diag}| vs sum|off|={off_sum} {status}")

print(f"\n--- Gauss-Seidel (es = 5%) ---")
x = gauss_seidel(A, b, tol=5.0)

x_exact = np.linalg.solve(A, b)
print(f"\nSolusi Gauss-Seidel: c1={x[0]:.4f}, c2={x[1]:.4f}, c3={x[2]:.4f}")
print(f"Solusi eksak:        c1={x_exact[0]:.4f}, c2={x_exact[1]:.4f}, c3={x_exact[2]:.4f}")
