"""
Soal 11.12 - Gauss-Seidel (a) tanpa relaxation, (b) dengan underrelaxation (λ=0.95)
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def gauss_seidel_sor(A, b, omega=1.0, tol=5.0, max_iter=300, verbose=True):
    """Gauss-Seidel + SOR (Successive Over/Under-Relaxation)."""
    n = len(b)
    x = np.zeros(n)

    if verbose:
        print(f"{'Iter':>5} | " + " ".join([f"{'x'+str(i+1):>10}" for i in range(n)]) + f" | {'Max %err':>10}")
        print("-" * 60)

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
            print(f"{iteration+1:>5} | " + " ".join([f"{xi:>10.5f}" for xi in x]) + f" | {max_ea:>10.4f}%")

        if max_ea < tol:
            if verbose:
                print(f"\n✓ Konvergen pada iterasi {iteration+1}")
            return x, iteration + 1

    if verbose:
        print(f"\n⚠ Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter

# Sistem dari soal 11.12 (mungkin tidak diagonally dominant → perlu susun ulang)
# -2*x1 + 12*x2 = 40
# 6*x1 - x2 = 15
A_orig = np.array([
    [-2.0, 12.0],
    [ 6.0, -1.0]
], dtype=float)
b_orig = np.array([40.0, 15.0], dtype=float)

print("=" * 60)
print("Soal 11.12 - Gauss-Seidel tanpa dan dengan Underrelaxation (λ=0.95)")
print("=" * 60)
print("\nSistem asli:")
print(f"  -2*x1 + 12*x2 = 40")
print(f"   6*x1 - x2    = 15")

# Cek dominansi - perlu disusun ulang
print("\nCek dominansi diagonal (sistem asli):")
for i in range(2):
    d = abs(A_orig[i][i])
    o = sum(abs(A_orig[i][j]) for j in range(2) if j!=i)
    print(f"  Baris {i+1}: |{d}| vs |{o}| {'✓' if d>=o else '✗ Perlu swap!'}")

# Susun ulang: swap baris
A = np.array([
    [ 6.0, -1.0],
    [-2.0, 12.0]
], dtype=float)
b = np.array([15.0, 40.0], dtype=float)

print("\nSetelah disusun ulang (baris 1 & 2 ditukar):")
print(f"   6*x1 - x2    = 15")
print(f"  -2*x1 + 12*x2 = 40")

print("\nCek dominansi (setelah swap):")
for i in range(2):
    d = abs(A[i][i])
    o = sum(abs(A[i][j]) for j in range(2) if j!=i)
    print(f"  Baris {i+1}: |{d}| vs |{o}| {'✓' if d>=o else '✗'}")

print("\n--- (a) Gauss-Seidel tanpa Relaxation (λ=1.0) ---")
x1, it1 = gauss_seidel_sor(A, b, omega=1.0, tol=5.0)
print(f"Solusi: x1={x1[0]:.5f}, x2={x1[1]:.5f} (iterasi: {it1})")

print("\n--- (b) Gauss-Seidel dengan Underrelaxation (λ=0.95) ---")
x2, it2 = gauss_seidel_sor(A, b, omega=0.95, tol=5.0)
print(f"Solusi: x1={x2[0]:.5f}, x2={x2[1]:.5f} (iterasi: {it2})")

x_exact = np.linalg.solve(A, b)
print(f"\nSolusi eksak: x1={x_exact[0]:.5f}, x2={x_exact[1]:.5f}")
