"""
Soal 11.23 - Analisis Operasi: Thomas Algorithm vs Eliminasi Gauss
Sumber: Numerical Methods for Engineers, Chapra & Canale
"""
import numpy as np
import matplotlib.pyplot as plt

def count_thomas_operations(n):
    """
    Menghitung jumlah operasi Thomas Algorithm.
    
    Forward sweep: (n-1) pembagian + (n-1) perkalian + (n-1) pengurangan = 3(n-1) ops
    Back substitution: 1 pembagian + (n-1)*(1 perkalian + 1 pengurangan + 1 pembagian) = 1 + 3(n-1) ops
    Total: ~5n - 4 operasi
    """
    # Forward sweep
    forward_div = n - 1
    forward_mul = n - 1
    forward_sub = n - 1
    
    # Back substitution
    back_div = n
    back_mul = n - 1
    back_sub = n - 1
    
    total = forward_div + forward_mul + forward_sub + back_div + back_mul + back_sub
    return total

def count_gauss_operations(n):
    """
    Jumlah operasi Eliminasi Gauss tanpa pivoting.
    Forward elimination: ~(2/3)n^3 + O(n^2)
    Back substitution:   ~n^2 + O(n)
    Total: ~(2/3)n^3 + n^2 - (2/3)n (pendekatan)
    Dari buku Chapra Sec 9.2.1:
    Total = (2n^3 + 3n^2 - 5n)/3 (termasuk forward + back)
    """
    return (2 * n**3 + 3 * n**2 - 5 * n) / 3

print("=" * 60)
print("Soal 11.23 - Analisis Operasi: Thomas vs Gauss Elimination")
print("=" * 60)

print("\nFormula:")
print("  Thomas Algorithm:    ~5n - 4 operasi")
print("  Gauss Elimination:   ~(2n³ + 3n² - 5n) / 3 operasi")

# Tabel perbandingan
print(f"\n{'n':>5} | {'Thomas':>12} | {'Gauss':>15} | {'Rasio (Gauss/Thomas)':>20}")
print("-" * 60)
ns = range(2, 21)
thomas_ops = []
gauss_ops = []
for n in ns:
    t_ops = count_thomas_operations(n)
    g_ops = count_gauss_operations(n)
    ratio = g_ops / t_ops
    thomas_ops.append(t_ops)
    gauss_ops.append(g_ops)
    print(f"{n:>5} | {t_ops:>12.0f} | {g_ops:>15.1f} | {ratio:>20.1f}x")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Absolut operasi
ax1 = axes[0]
ns_arr = np.array(list(ns))
ax1.plot(ns_arr, thomas_ops, 'b-o', markersize=6, linewidth=2, label='Thomas Algorithm')
ax1.plot(ns_arr, gauss_ops, 'r-s', markersize=6, linewidth=2, label='Gauss Elimination')
ax1.set_xlabel('Ukuran sistem n')
ax1.set_ylabel('Jumlah operasi')
ax1.set_title('Perbandingan Jumlah Operasi')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Rasio
ratios = [g/t for g, t in zip(gauss_ops, thomas_ops)]
ax2 = axes[1]
ax2.plot(ns_arr, ratios, 'g-^', markersize=6, linewidth=2)
ax2.fill_between(ns_arr, ratios, alpha=0.3, color='green')
ax2.set_xlabel('Ukuran sistem n')
ax2.set_ylabel('Rasio Operasi (Gauss / Thomas)')
ax2.set_title('Efisiensi Relatif Thomas vs Gauss')
ax2.grid(True, alpha=0.3)
for xi, yi in zip(ns_arr[::3], ratios[::3]):
    ax2.annotate(f'{yi:.0f}x', (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

plt.tight_layout()
plt.savefig('soal11_23_operations.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n✓ Plot disimpan: soal11_23_operations.png")
print(f"\nKesimpulan: Untuk n=20, Thomas memerlukan {count_thomas_operations(20)} operasi")
print(f"sedangkan Gauss memerlukan {count_gauss_operations(20):.0f} operasi")
print(f"→ Thomas ~{count_gauss_operations(20)/count_thomas_operations(20):.0f}x lebih efisien!")
