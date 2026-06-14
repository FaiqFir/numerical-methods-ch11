"""
Soal 11.13 - Gauss-Seidel (a) tanpa relaxation, (b) dengan overrelaxation (λ=1.2)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Sistem diagonally dominant setelah disusun ulang:
  -3x1 + x2 - x3 = 0   --> disusun ulang agar diagonal dominan
"""
import numpy as np

def gauss_seidel_sor(A, b, omega=1.0, tol=5.0, max_iter=100, verbose=True):
    n = len(b)
    x = np.zeros(n)
    if verbose:
        print(f"{'Iter':>5} | " + " ".join([f"{'x'+str(i+1):>12}" for i in range(n)]) + f" | {'Max %err':>10}")
        print("-" * 65)
    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_gs = (b[i] - sigma) / A[i][i]
            x[i] = omega * x_gs + (1 - omega) * x[i]
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-15:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100
        max_ea = np.max(ea)
        if verbose:
            print(f"{iteration+1:>5} | " + " ".join([f"{xi:>12.6f}" for xi in x]) + f" | {max_ea:>10.4f}%")
        if max_ea < tol:
            if verbose:
                print(f"\n✓ Konvergen pada iterasi {iteration+1}")
            return x, iteration + 1
    if verbose:
        print(f"\n⚠ Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter

print("=" * 65)
print("Soal 11.13 - Gauss-Seidel tanpa dan dengan Overrelaxation (λ=1.2)")
print("=" * 65)

# Sistem 3x3 dari konteks buku Chapra soal 11.13
# Setelah disusun ulang agar diagonally dominant:
# 8x1 - 3x2 + 2x3 = 20
# 4x1 + 11x2 - x3 = 33
# 6x1 + 3x2 + 12x3 = 36
A = np.array([
    [ 8.0, -3.0,  2.0],
    [ 4.0, 11.0, -1.0],
    [ 6.0,  3.0, 12.0]
], dtype=float)
b = np.array([20.0, 33.0, 36.0], dtype=float)

print("\nSistem persamaan (setelah disusun ulang):")
print("  8x1 - 3x2 + 2x3 = 20")
print("  4x1 + 11x2 - x3 = 33")
print("  6x1 + 3x2 + 12x3 = 36")

print("\nCek dominansi diagonal:")
for i in range(3):
    d = abs(A[i][i])
    o = sum(abs(A[i][j]) for j in range(3) if j != i)
    print(f"  Baris {i+1}: |{d}| > {o}? {'✓' if d > o else '✗'}")

print("\n--- (a) Gauss-Seidel tanpa Relaxation (λ=1.0) ---")
x1, it1 = gauss_seidel_sor(A, b, omega=1.0, tol=5.0)
print(f"\nSolusi (λ=1.0): x1={x1[0]:.5f}, x2={x1[1]:.5f}, x3={x1[2]:.5f} ({it1} iterasi)")

print("\n--- (b) Gauss-Seidel dengan Overrelaxation (λ=1.2) ---")
x2, it2 = gauss_seidel_sor(A, b, omega=1.2, tol=5.0)
print(f"\nSolusi (λ=1.2): x1={x2[0]:.5f}, x2={x2[1]:.5f}, x3={x2[2]:.5f} ({it2} iterasi)")

x_exact = np.linalg.solve(A, b)
print(f"\nSolusi eksak:   x1={x_exact[0]:.5f}, x2={x_exact[1]:.5f}, x3={x_exact[2]:.5f}")
print(f"\nKesimpulan: λ=1.0 butuh {it1} iterasi, λ=1.2 butuh {it2} iterasi")
