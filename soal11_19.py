"""
Soal 11.19 - Condition Number Hilbert Matrix 10 Dimensi
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np
from scipy.linalg import hilbert

print("=" * 60)
print("Soal 11.19 - Hilbert Matrix 10 Dimensi: Condition Number")
print("=" * 60)

n = 10

# Buat Hilbert matrix 10x10
# H[i,j] = 1/(i+j+1)
H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])

print(f"\nHilbert matrix {n}x{n} (5 baris pertama):")
print(np.round(H[:5, :5], 6))

# Condition number (spectral = 2-norm)
cond_spectral = np.linalg.cond(H)
print(f"\nSpectral condition number: {cond_spectral:.6e}")
print(f"log10(cond): {np.log10(cond_spectral):.2f}")
print(f"Estimasi digit presisi hilang: ~{np.log10(cond_spectral):.1f} digit")

# Misalkan presisi mesin ~ 15-16 digit (float64)
digits_machine = 15.6  # log10(1/machine_epsilon) ≈ 15.6
digits_lost = np.log10(cond_spectral)
digits_reliable = digits_machine - digits_lost
print(f"\nPresisi mesin (float64): ~{digits_machine:.1f} digit")
print(f"Digit yang hilang karena ill-conditioning: ~{digits_lost:.1f}")
print(f"Digit yang masih dapat dipercaya: ~{digits_reliable:.1f}")

# Buat b sehingga solusi eksak = [1,1,...,1]
x_exact = np.ones(n)
b = H @ x_exact
print(f"\nVector b (b = H * [1,1,...,1]):")
print(np.round(b, 8))

# Hitung solusi numerik
x_num = np.linalg.solve(H, b)
print(f"\nSolusi numerik x:")
print(x_num)

# Error
error = x_num - x_exact
abs_error = np.abs(error)
print(f"\nError (x_num - x_exact):")
print(error)
print(f"\nMax absolute error: {np.max(abs_error):.6e}")
print(f"Mean absolute error: {np.mean(abs_error):.6e}")

# Bandingkan error yang diharapkan vs aktual
expected_error_order = cond_spectral * np.finfo(float).eps
print(f"\nError yang diharapkan (cond * eps): {expected_error_order:.6e}")
print(f"Error aktual (max): {np.max(abs_error):.6e}")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Nilai error per komponen
axes[0].bar(range(1, n+1), abs_error, color='steelblue', edgecolor='navy')
axes[0].set_xlabel('Indeks x')
axes[0].set_ylabel('|Error|')
axes[0].set_title(f'Error per Komponen\nHilbert {n}x{n}')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Plot 2: Condition number vs dimensi n
dims = range(2, 13)
conds = []
for d in dims:
    Hd = np.array([[1.0/(i+j+1) for j in range(d)] for i in range(d)])
    conds.append(np.linalg.cond(Hd))

axes[1].semilogy(dims, conds, 'r-o', markersize=6)
axes[1].axvline(10, color='g', linestyle='--', label='n=10')
axes[1].set_xlabel('Dimensi n')
axes[1].set_ylabel('Condition Number (log scale)')
axes[1].set_title('Condition Number Hilbert Matrix vs n')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('soal11_19_hilbert.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Plot disimpan: soal11_19_hilbert.png")
