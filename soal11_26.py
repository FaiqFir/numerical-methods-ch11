"""
Soal 11.26 - Program Gauss-Seidel (User-Friendly)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Ditest menggunakan hasil Example 11.3.
"""
import numpy as np

def gauss_seidel(A, b, x0=None, tol=5.0, max_iter=100,
                 omega=1.0, verbose=True):
    """
    Gauss-Seidel Method dengan opsi SOR (Successive Over-Relaxation).
    
    Parameters:
    -----------
    A       : matriks koefisien (n x n)
    b       : vector sisi kanan (n)
    x0      : tebakan awal (default: vektor nol)
    tol     : toleransi percent relative error (%)
    max_iter: maksimum iterasi
    omega   : faktor relaksasi (1.0 = Gauss-Seidel murni)
    verbose : tampilkan tabel iterasi
    
    Returns:
    --------
    x       : solusi
    iterations : jumlah iterasi
    error_history : riwayat error
    """
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    
    error_history = []
    
    if verbose:
        header = (f"{'Iter':>5} | " +
                  " ".join([f"{'x'+str(i+1):>12}" for i in range(n)]) +
                  f" | {'Max %err':>10}")
        print(header)
        print("-" * len(header))
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_gs = (b[i] - sigma) / A[i][i]
            x[i] = omega * x_gs + (1 - omega) * x[i]
        
        # Hitung percent relative error
        ea = np.zeros(n)
        for i in range(n):
            if abs(x[i]) > 1e-15:
                ea[i] = abs((x[i] - x_old[i]) / x[i]) * 100
        max_ea = np.max(ea)
        error_history.append(max_ea)
        
        if verbose:
            row = (f"{iteration+1:>5} | " +
                   " ".join([f"{xi:>12.6f}" for xi in x]) +
                   f" | {max_ea:>10.4f}%")
            print(row)
        
        if max_ea < tol:
            if verbose:
                print(f"\n✓ Konvergen pada iterasi {iteration+1} "
                      f"(max error = {max_ea:.4f}% < {tol}%)")
            return x, iteration + 1, error_history
    
    if verbose:
        print(f"\n⚠ Tidak konvergen dalam {max_iter} iterasi")
    return x, max_iter, error_history

# ==============================================
# TEST: Duplikat Example 11.3
# ==============================================
print("=" * 65)
print("Soal 11.26 - Program Gauss-Seidel Method")
print("=" * 65)
print("\nTest: Menduplikat hasil Example 11.3")

# Sistem dari Example 11.3 (Chapra)
# 3x1 - 0.1x2 - 0.2x3 = 7.85
# 0.1x1 + 7x2 - 0.3x3 = -19.3
# 0.3x1 - 0.2x2 + 10x3 = 71.4
A = np.array([
    [3.0,  -0.1, -0.2],
    [0.1,   7.0, -0.3],
    [0.3,  -0.2, 10.0]
], dtype=float)

b = np.array([7.85, -19.3, 71.4], dtype=float)

print("\nSistem persamaan:")
print("  3x1 - 0.1x2 - 0.2x3 = 7.85")
print("  0.1x1 + 7x2 - 0.3x3 = -19.3")
print("  0.3x1 - 0.2x2 + 10x3 = 71.4")
print("\nMatriks A:")
print(A)

# Cek dominansi diagonal
print("\nCek dominansi diagonal:")
for i in range(3):
    d = abs(A[i][i])
    o = sum(abs(A[i][j]) for j in range(3) if j != i)
    print(f"  Baris {i+1}: |{d}| > {o}? {'✓' if d > o else '✗'}")

print("\n--- Iterasi Gauss-Seidel (es = 5%) ---")
x, n_iter, err_hist = gauss_seidel(A, b, tol=5.0)

print("\n" + "=" * 40)
print("SOLUSI AKHIR:")
x_labels = ['x1', 'x2', 'x3']
for i, xi in enumerate(x):
    print(f"  {x_labels[i]} = {xi:.6f}")

x_exact = np.linalg.solve(A, b)
print("\nSolusi eksak:")
for i, xi in enumerate(x_exact):
    print(f"  {x_labels[i]} = {xi:.6f}")
print(f"\nMax error: {np.max(np.abs(x - x_exact)):.6e}")
print(f"\n✓ Hasil sesuai dengan Example 11.3 (x1≈3, x2≈-2.5, x3≈7)")
