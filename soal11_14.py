"""
Soal 11.14 - Gambar Ulang Fig. 11.5 untuk Slope 1 dan -1
Sumber: Numerical Methods for Engineers, Chapra & Canale

Pertanyaan: Gambar ulang Fig. 11.5 untuk kasus slope = 1 dan -1.
Apa hasil penerapan Gauss-Seidel pada sistem tersebut?

Sistem:
  x1 + x2 = 3   (slope = -1)
  x1 - x2 = 1   (slope = +1)
  Solusi eksak: x1=2, x2=1
"""
import numpy as np
import matplotlib.pyplot as plt

print("=" * 55)
print("Soal 11.14 - Gambar Ulang Fig. 11.5 (slope +1 dan -1)")
print("=" * 55)

A = np.array([[1.0,  1.0],
              [1.0, -1.0]], dtype=float)
b = np.array([3.0, 1.0], dtype=float)

print("\nSistem persamaan:")
print("  x1 + x2 = 3  (slope = -1)")
print("  x1 - x2 = 1  (slope = +1)")
print("  Solusi eksak: x1=2, x2=1")

print("\nCek dominansi diagonal:")
for i in range(2):
    d = abs(A[i][i])
    o = sum(abs(A[i][j]) for j in range(2) if j != i)
    status = "✓ dominan" if d > o else "✗ TIDAK dominan"
    print(f"  Baris {i+1}: |{d}| vs |{o}| → {status}")

# Jalankan Gauss-Seidel dan rekam jalur iterasi
x = np.array([0.0, 0.0])
path = [x.copy()]
max_iter = 12

print("\n--- Iterasi Gauss-Seidel ---")
print(f"{'Iter':>5} | {'x1':>10} | {'x2':>10} | Status")
print("-" * 42)

for k in range(max_iter):
    x_old = x.copy()
    x[0] = (b[0] - A[0][1] * x[1]) / A[0][0]
    x[1] = (b[1] - A[1][0] * x[0]) / A[1][1]
    path.append(x.copy())
    # Cek divergensi
    diverging = np.any(np.abs(x) > 1e6)
    status = "DIVERGEN!" if diverging else ""
    print(f"{k+1:>5} | {x[0]:>10.4f} | {x[1]:>10.4f} | {status}")
    if diverging:
        break

path = np.array(path)

# ============================================================
# PLOT - Gambar ulang Fig. 11.5
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Garis persamaan + jalur iterasi ---
ax = axes[0]
x_range = np.linspace(-1, 5, 200)
y1 = 3 - x_range          # x1 + x2 = 3  → slope -1
y2 = x_range - 1          # x1 - x2 = 1  → slope +1

ax.plot(x_range, y1, 'b-', linewidth=2.5, label='$x_1 + x_2 = 3$ (slope = −1)')
ax.plot(x_range, y2, 'r-', linewidth=2.5, label='$x_1 − x_2 = 1$ (slope = +1)')
ax.plot(2, 1, 'g*', markersize=18, zorder=5, label='Solusi eksak (2, 1)')

# Jalur iterasi Gauss-Seidel (zig-zag)
n_show = min(len(path), 8)
for k in range(n_show - 1):
    ax.annotate('', xy=path[k+1], xytext=path[k],
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
ax.plot(path[:n_show, 0], path[:n_show, 1], 'o--',
        color='purple', markersize=6, linewidth=1, alpha=0.7, label='Jalur Gauss-Seidel')
ax.plot(path[0, 0], path[0, 1], 'ks', markersize=10, label='Titik awal (0, 0)')

ax.set_xlim(-1, 5)
ax.set_ylim(-2, 4)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Fig. 11.5 (Slope +1 dan −1)\nGauss-Seidel Tidak Konvergen', fontsize=11)

# --- Plot 2: Nilai x1 dan x2 vs iterasi ---
ax2 = axes[1]
iters = range(len(path))
ax2.plot(iters, path[:, 0], 'b-o', markersize=7, linewidth=2, label='$x_1$')
ax2.plot(iters, path[:, 1], 'r-s', markersize=7, linewidth=2, label='$x_2$')
ax2.axhline(2, color='b', linestyle='--', alpha=0.5, linewidth=1, label='$x_1 = 2$ (eksak)')
ax2.axhline(1, color='r', linestyle='--', alpha=0.5, linewidth=1, label='$x_2 = 1$ (eksak)')
ax2.set_xlabel('Iterasi', fontsize=12)
ax2.set_ylabel('Nilai', fontsize=12)
ax2.set_title('Osilasi / Divergensi x vs Iterasi', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('soal11_14_fig115.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n✓ Plot disimpan: soal11_14_fig115.png")
print("\nKesimpulan:")
print("  Ketika slope kedua persamaan adalah +1 dan -1,")
print("  matriks tidak diagonally dominant (|1| = |1|).")
print("  Gauss-Seidel menghasilkan osilasi permanen dan")
print("  TIDAK konvergen ke solusi eksak (2, 1).")
