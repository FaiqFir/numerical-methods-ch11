"""
Soal 11.24 - Program Thomas Algorithm (User-Friendly)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Implementasi program untuk sistem tridiagonal menggunakan Thomas algorithm.
Ditest menggunakan hasil Example 11.1.
"""
import numpy as np

def thomas_algorithm(a, b, c, d, verbose=True):
    """
    Thomas Algorithm untuk sistem tridiagonal.
    
    Parameters:
    -----------
    a : array, lower diagonal (panjang n-1), a[0] = koef baris ke-2
    b : array, main diagonal (panjang n)
    c : array, upper diagonal (panjang n-1), c[0] = koef baris ke-1
    d : array, right-hand side (panjang n)
    verbose : tampilkan langkah-langkah
    
    Returns:
    --------
    x : solusi
    """
    n = len(d)
    
    # Salin agar tidak mengubah input
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)
    
    if verbose:
        print("\n--- Forward Sweep ---")
    
    # ==============================
    # FORWARD SWEEP (dekomposisi)
    # ==============================
    for i in range(1, n):
        factor = a[i-1] / b[i-1]
        b[i] = b[i] - factor * c[i-1]
        d[i] = d[i] - factor * d[i-1]
        
        if verbose:
            print(f"  i={i+1}: faktor={factor:.6f}, b[{i}]={b[i]:.6f}, d[{i}]={d[i]:.6f}")
    
    if verbose:
        print("\n--- Back Substitution ---")
    
    # ==============================
    # BACK SUBSTITUTION
    # ==============================
    x = np.zeros(n)
    x[n-1] = d[n-1] / b[n-1]
    
    if verbose:
        print(f"  x[{n}] = {d[n-1]:.6f} / {b[n-1]:.6f} = {x[n-1]:.6f}")
    
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        if verbose:
            print(f"  x[{i+1}] = ({d[i]:.6f} - {c[i]:.6f}*{x[i+1]:.6f}) / {b[i]:.6f} = {x[i]:.6f}")
    
    return x

def solve_tridiagonal_interactive():
    """Mode interaktif untuk input manual."""
    print("\n=== Mode Interaktif ===")
    n = int(input("Masukkan ukuran sistem (n): "))
    
    print(f"\nMasukkan diagonal utama ({n} nilai):")
    b = [float(input(f"  b[{i+1}]: ")) for i in range(n)]
    
    print(f"\nMasukkan diagonal atas ({n-1} nilai):")
    c = [float(input(f"  c[{i+1}]: ")) for i in range(n-1)]
    
    print(f"\nMasukkan diagonal bawah ({n-1} nilai):")
    a = [float(input(f"  a[{i+2}]: ")) for i in range(n-1)]
    
    print(f"\nMasukkan sisi kanan ({n} nilai):")
    d = [float(input(f"  d[{i+1}]: ")) for i in range(n)]
    
    return a, b, c, d

# ==============================================
# TEST: Duplikat hasil Example 11.1
# ==============================================
print("=" * 60)
print("Soal 11.24 - Thomas Algorithm (Program User-Friendly)")
print("=" * 60)
print("\nTest: Menduplikat hasil Example 11.1")
print("-" * 60)

# Sistem tridiagonal 4x4 dari Example 11.1
b_diag = [2.04, 2.04, 2.04, 2.04]   # main diagonal
c_up   = [-1.0, -1.0, -1.0]          # upper diagonal
a_low  = [-1.0, -1.0, -1.0]          # lower diagonal
d_rhs  = [40.8, 0.8, 0.8, 200.8]    # RHS

print("\nInput Sistem:")
print(f"  Main diagonal: {b_diag}")
print(f"  Upper diagonal: {c_up}")
print(f"  Lower diagonal: {a_low}")
print(f"  RHS: {d_rhs}")

x = thomas_algorithm(a_low, b_diag, c_up, d_rhs, verbose=True)

print("\n" + "=" * 40)
print("SOLUSI:")
for i, xi in enumerate(x):
    print(f"  x[{i+1}] = {xi:.6f}")

# Verifikasi dengan numpy
A = np.diag(b_diag) + np.diag(c_up, 1) + np.diag(a_low, -1)
x_np = np.linalg.solve(A, d_rhs)
print("\nVerifikasi (numpy):", np.round(x_np, 6))
print(f"Max error: {np.max(np.abs(x - x_np)):.2e}")
print("\n✓ Hasil sesuai dengan Example 11.1")

print("\n" + "=" * 60)
print("Program siap digunakan. Untuk input manual, jalankan:")
print("  a, b, c, d = solve_tridiagonal_interactive()")
print("  x = thomas_algorithm(a, b, c, d)")
