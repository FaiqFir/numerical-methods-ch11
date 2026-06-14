"""
Soal 11.10 - Jacobi Iteration untuk Sistem Reaktor (Prob. 11.9)
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def jacobi_iteration(A, b, tol=5.0, max_iter=200):
    """
    Jacobi Iteration.
    Perbedaan dari Gauss-Seidel: update semua x sekaligus menggunakan nilai lama.
    """
    n = len(b)
    x = np.zeros(n)

    print(f"{'Iter':>5} | " + " ".join([f"{'c'+str(i+1):>12}" for i in range(n)]) + f" | {'Max %err':>10}")
    print("-" * (5 + 14*n + 15))

    for iteration in range(max_iter):
        x_old = x.copy()
        x_new = np.zeros(n)

        for i in range(n):
            sigma = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        x = x_new.copy()

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
    else:
        print(f"\n⚠ Tidak konvergen dalam {max_iter} iterasi")

    return x

# Sistem dari Prob. 11.9 (sama)
A = np.array([
    [15.0, -3.0, -1.0],
    [-3.0,  18.0, -6.0],
    [-4.0,  -1.0, 12.0]
], dtype=float)

b = np.array([3800.0, 1200.0, 2350.0], dtype=float)

print("=" * 60)
print("Soal 11.10 - Jacobi Iteration: Reaktor Tersambung")
print("=" * 60)
print("\nSistem persamaan:")
for i in range(3):
    print(f"  {A[i][0]}*c1 + {A[i][1]}*c2 + {A[i][2]}*c3 = {b[i]}")

print("\n--- Jacobi Iteration (es = 5%) ---")
x_jacobi = jacobi_iteration(A, b, tol=5.0)

# Gauss-Seidel untuk perbandingan
print("\n--- Perbandingan dengan Gauss-Seidel ---")
def gauss_seidel_simple(A, b, tol=5.0, max_iter=200):
    n = len(b)
    x = np.zeros(n)
    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]
        ea = [abs((x[i]-x_old[i])/x[i])*100 if abs(x[i])>1e-15 else 0 for i in range(n)]
        if max(ea) < tol:
            return x, iteration+1
    return x, max_iter

x_gs, iter_gs = gauss_seidel_simple(A, b)

x_exact = np.linalg.solve(A, b)
print(f"\nSolusi Jacobi:       c1={x_jacobi[0]:.4f}, c2={x_jacobi[1]:.4f}, c3={x_jacobi[2]:.4f}")
print(f"Solusi Gauss-Seidel: c1={x_gs[0]:.4f}, c2={x_gs[1]:.4f}, c3={x_gs[2]:.4f} (iter={iter_gs})")
print(f"Solusi eksak:        c1={x_exact[0]:.4f}, c2={x_exact[1]:.4f}, c3={x_exact[2]:.4f}")
print("\nKesimpulan: Gauss-Seidel biasanya lebih cepat konvergen dari Jacobi.")
