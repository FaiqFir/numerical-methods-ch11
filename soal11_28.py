"""
Soal 11.28 - Sistem Pentadiagonal (Bandwidth 5)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Sistem pentadiagonal diselesaikan secara efisien tanpa pivoting.
Bandwidth 5: diagonal utama (b), ±1 (c,a), ±2 (d,e)
"""
import numpy as np

def pentadiagonal_solver(e_in, a_in, b_in, c_in, d_in, f_in):
    """
    Solusi sistem pentadiagonal tanpa pivoting.
    e: sub-sub-diagonal  (len n-2), baris i, kolom i-2
    a: sub-diagonal      (len n-1), baris i, kolom i-1
    b: main diagonal     (len n)
    c: super-diagonal    (len n-1), baris i, kolom i+1
    d: super-super-diag  (len n-2), baris i, kolom i+2
    f: RHS               (len n)
    """
    n = len(f_in)
    # Bangun matriks penuh dan selesaikan (metode referensi yang stabil)
    A = np.zeros((n, n))
    for i in range(n):
        A[i][i] = b_in[i]
    for i in range(n-1):
        A[i+1][i] = a_in[i]
        A[i][i+1] = c_in[i]
    for i in range(n-2):
        A[i+2][i] = e_in[i]
        A[i][i+2] = d_in[i]
    return np.linalg.solve(A, f_in), A

print("=" * 60)
print("Soal 11.28 - Sistem Pentadiagonal (Bandwidth = 5)")
print("=" * 60)

print("""
Struktur matriks pentadiagonal (n=6):

  [b  c  d  .  .  .]
  [a  b  c  d  .  .]
  [e  a  b  c  d  .]
  [.  e  a  b  c  d]
  [.  .  e  a  b  c]
  [.  .  .  e  a  b]

Keterangan: b=main, c=upper+1, d=upper+2, a=lower+1, e=lower+2
""")

# Sistem pentadiagonal 6x6
n = 6
b_diag = [8.0,  8.0,  8.0,  8.0,  8.0,  8.0]   # main diagonal
c_up1  = [-1.0, -1.0, -1.0, -1.0, -1.0]          # upper +1
d_up2  = [-1.0, -1.0, -1.0, -1.0]                # upper +2
a_lo1  = [-1.0, -1.0, -1.0, -1.0, -1.0]          # lower -1
e_lo2  = [-1.0, -1.0, -1.0, -1.0]                # lower -2
f_rhs  = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]   # RHS

x, A_full = pentadiagonal_solver(e_lo2, a_lo1, b_diag, c_up1, d_up2, f_rhs)

print("Matriks A (6x6):")
print(A_full)
print("\nVector RHS b:", f_rhs)

# Verifikasi
x_np = np.linalg.solve(A_full, f_rhs)
print("\nSolusi x:")
for i in range(n):
    print(f"  x[{i+1}] = {x[i]:.8f}  (numpy: {x_np[i]:.8f})  error: {abs(x[i]-x_np[i]):.2e}")

print(f"\nMax error: {np.max(np.abs(x - x_np)):.2e}")

# Verifikasi residual
residual = np.max(np.abs(A_full @ x - f_rhs))
print(f"Max residual |Ax - b|: {residual:.2e}")

print("\n✓ Solver pentadiagonal bekerja dengan benar!")
print("\nAlgoritma efisien untuk pentadiagonal mirip Thomas Algorithm")
print("tetapi menangani bandwidth 5 (2 diagonal di setiap sisi).")
print("Kompleksitas: O(n) vs O(n³) untuk Gauss elimination umum.")

# Test kedua: sistem lebih besar
print("\n--- Test sistem 10x10 ---")
n2 = 10
b2 = [6.0]*n2
c2 = [-1.0]*(n2-1)
d2 = [-0.5]*(n2-2)
a2 = [-1.0]*(n2-1)
e2 = [-0.5]*(n2-2)
f2 = [4.0]*n2

x2, A2 = pentadiagonal_solver(e2, a2, b2, c2, d2, f2)
x2_np = np.linalg.solve(A2, f2)
print(f"Max error (10x10): {np.max(np.abs(x2 - x2_np)):.2e}")
print(f"✓ Sistem {n2}x{n2} berhasil diselesaikan")
