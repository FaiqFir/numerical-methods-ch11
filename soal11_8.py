"""
Soal 11.8 - Gauss-Seidel dengan Overrelaxation (λ=1.2) untuk Sistem Tridiagonal
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np

def gauss_seidel_sor(A, b, x0=None, omega=1.0, tol=5.0, max_iter=100):
    """
    Gauss-Seidel dengan Successive Over-Relaxation (SOR).
    omega = 1.0: Gauss-Seidel biasa
    omega > 1.0: Over-relaxation
    omega < 1.0: Under-relaxation
    tol: toleransi percent relative error (%)
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    history = []
    print(f"\n{'Iter':>5} | {'x1':>12} {'x2':>12} {'x3':>12} {'x4':>12} | {'Max %err':>10}")
    print("-" * 75)
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new_gs = (b[i] - sigma) / A[i][i]
            # SOR update
            x[i] = omega * x_new_gs + (1 - omega) * x[i]
        
        # Hitung percent relative error
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-15:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100
        
        max_ea = np.max(ea)
        history.append((iteration + 1, x.copy(), max_ea))
        
        if len(x) == 4:
            print(f"{iteration+1:>5} | {x[0]:>12.6f} {x[1]:>12.6f} {x[2]:>12.6f} {x[3]:>12.6f} | {max_ea:>10.4f}%")
        else:
            print(f"{iteration+1:>5} | {' '.join([f'{xi:>12.6f}' for xi in x])} | {max_ea:>10.4f}%")
        
        if max_ea < tol:
            print(f"\n✓ Konvergen pada iterasi {iteration+1} (error < {tol}%)")
            break
    
    return x, history

# Sistem tridiagonal dari Prob. 11.1
A = np.array([
    [2.04, -1.0,  0.0,  0.0],
    [-1.0,  2.04, -1.0,  0.0],
    [0.0,  -1.0,  2.04, -1.0],
    [0.0,   0.0, -1.0,  2.04]
], dtype=float)

b = np.array([40.8, 0.8, 0.8, 200.8], dtype=float)

print("=" * 75)
print("Soal 11.8 - Gauss-Seidel dengan Overrelaxation (λ=1.2)")
print("=" * 75)
print("\nMatriks A:")
print(A)
print("Vector b:", b)

# (a) Tanpa overrelaxation (λ=1.0)
print("\n--- (a) Gauss-Seidel tanpa Overrelaxation (λ=1.0), es=5% ---")
x1, h1 = gauss_seidel_sor(A, b, omega=1.0, tol=5.0)
print(f"\nSolusi: {x1}")

# (b) Dengan overrelaxation (λ=1.2)
print("\n--- (b) Gauss-Seidel dengan Overrelaxation (λ=1.2), es=5% ---")
x2, h2 = gauss_seidel_sor(A, b, omega=1.2, tol=5.0)
print(f"\nSolusi: {x2}")

# Solusi eksak
x_exact = np.linalg.solve(A, b)
print(f"\nSolusi eksak: {x_exact}")
print(f"Error λ=1.0: {np.max(np.abs(x1 - x_exact)):.6f}")
print(f"Error λ=1.2: {np.max(np.abs(x2 - x_exact)):.6f}")
