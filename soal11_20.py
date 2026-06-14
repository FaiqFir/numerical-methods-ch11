"""
Soal 11.20 - Vandermonde Matrix 6 Dimensi: Condition Number
Sumber: Numerical Methods for Engineers, Chapra & Canale

x1=4, x2=2, x3=7, x4=10, x5=3, x6=5
Vandermonde matrix: V[i,j] = x_i^(n-1-j)
"""
import numpy as np
import matplotlib.pyplot as plt

def vandermonde_matrix(x):
    """Buat matriks Vandermonde dari vector x."""
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i][j] = x[i] ** (n - 1 - j)
    return V

print("=" * 60)
print("Soal 11.20 - Vandermonde Matrix 6 Dimensi")
print("=" * 60)

# Nilai x yang diberikan
x = np.array([4.0, 2.0, 7.0, 10.0, 3.0, 5.0])
n = len(x)

print(f"\nNilai x: {x}")
print(f"Dimensi: {n}x{n}")

# Buat Vandermonde matrix
V = vandermonde_matrix(x)
print("\nVandermonde Matrix V:")
print(np.round(V, 2))

# Condition number (spectral)
cond = np.linalg.cond(V)
print(f"\nSpectral condition number: {cond:.6e}")
print(f"log10(cond): {np.log10(cond):.2f}")
print(f"Estimasi digit presisi hilang: ~{np.log10(cond):.1f} digit")

# Solusi dengan b = V * [1,1,...,1]
x_exact = np.ones(n)
b = V @ x_exact
print(f"\nVector b (b = V * [1,1,...,1]):")
print(np.round(b, 6))

# Solusi numerik
x_num = np.linalg.solve(V, b)
print(f"\nSolusi numerik:")
print(x_num)

error = np.abs(x_num - x_exact)
print(f"\nError absolut:")
print(error)
print(f"\nMax error: {np.max(error):.6e}")

# Bandingkan dengan Hilbert (soal 11.19) 
H6 = np.array([[1.0/(i+j+1) for j in range(6)] for i in range(6)])
cond_hilbert6 = np.linalg.cond(H6)

print(f"\n--- Perbandingan ---")
print(f"Vandermonde 6x6 cond: {cond:.6e}")
print(f"Hilbert 6x6 cond:     {cond_hilbert6:.6e}")
print(f"Vandermonde {'lebih' if cond > cond_hilbert6 else 'kurang'} ill-conditioned dari Hilbert")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(range(1, n+1), error, color='tomato', edgecolor='darkred')
axes[0].set_xlabel('Indeks x')
axes[0].set_ylabel('|Error|')
axes[0].set_title('Error per Komponen\nVandermonde 6x6')
if np.max(error) > 0:
    axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Log scale perbandingan
categories = ['Vandermonde\n6x6', 'Hilbert\n6x6']
values = [cond, cond_hilbert6]
axes[1].bar(categories, values, color=['tomato', 'steelblue'], edgecolor='black')
axes[1].set_ylabel('Condition Number (log scale)')
axes[1].set_yscale('log')
axes[1].set_title('Perbandingan Condition Number')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('soal11_20_vandermonde.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Plot disimpan: soal11_20_vandermonde.png")
