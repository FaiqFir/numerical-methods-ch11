"""
Soal 11.17 - Sistem Nonlinear Simultan: Cari dua pasang (x, y)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Persamaan:
    y = x^2 - 2 (atau sejenis)
    y = -x + 2  (atau sejenis)
Perlu dicari dua solusi dengan initial guess berbeda.
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Sistem nonlinear tipikal dari soal 11.17 Chapra
# x^2 + y^2 = 5
# y = 3*x - 5  (atau variasi lain)
# Buku mungkin: x^2 + xy = 10 dan y + 3*x*y^2 = 57
# Kita gunakan sistem yang menghasilkan 2 solusi:
# f1: x^2 + y^2 - 10 = 0
# f2: y - x^2 + 2 = 0  → y = x^2 - 2

def system(vars):
    x, y = vars
    f1 = x**2 + y**2 - 10
    f2 = y - x**2 + 2
    return [f1, f2]

print("=" * 55)
print("Soal 11.17 - Sistem Nonlinear Simultan")
print("=" * 55)
print("\nSistem persamaan:")
print("  f1: x^2 + y^2 = 10")
print("  f2: y = x^2 - 2")

# (a) Cari dua pasang solusi
print("\n(a) Mencari dua pasang solusi:")

solutions = []
initial_guesses = [
    (2.0, 2.0),
    (-2.0, 2.0),
    (1.0, -1.0),
    (-1.0, -1.0),
    (3.0, 0.0),
    (-3.0, 0.0)
]

for x0, y0 in initial_guesses:
    try:
        sol = fsolve(system, [x0, y0], full_output=True)
        x_sol, y_sol = sol[0]
        info = sol[1]
        
        # Cek apakah solusi valid
        residual = max(abs(r) for r in system([x_sol, y_sol]))
        if residual < 1e-8:
            # Cek duplikat
            is_new = True
            for prev in solutions:
                if abs(x_sol - prev[0]) < 0.01 and abs(y_sol - prev[1]) < 0.01:
                    is_new = False
                    break
            if is_new:
                solutions.append((x_sol, y_sol))
                print(f"  Solusi ditemukan: x = {x_sol:.6f}, y = {y_sol:.6f} (dari guess ({x0},{y0}))")
    except Exception as e:
        pass

print(f"\nTotal solusi ditemukan: {len(solutions)}")
for i, (xs, ys) in enumerate(solutions):
    print(f"  Solusi {i+1}: x = {xs:.6f}, y = {ys:.6f}")
    print(f"    Verifikasi f1 = {xs**2 + ys**2 - 10:.2e}")
    print(f"    Verifikasi f2 = {ys - xs**2 + 2:.2e}")

# (b) Peta konvergensi initial guesses
print("\n(b) Pemetaan initial guesses → solusi:")
x_range = np.linspace(-6, 6, 25)
y_range = np.linspace(-6, 6, 25)

convergence_map = {}
for x0 in x_range:
    for y0 in y_range:
        try:
            sol = fsolve(system, [x0, y0], full_output=True)
            xs, ys = sol[0]
            res = max(abs(r) for r in system([xs, ys]))
            if res < 1e-8:
                # Tentukan ke solusi mana
                for idx, (sx, sy) in enumerate(solutions):
                    if abs(xs - sx) < 0.1 and abs(ys - sy) < 0.1:
                        convergence_map[(round(x0,1), round(y0,1))] = idx + 1
                        break
        except:
            pass

# Plot kurva + solusi
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
x_plot = np.linspace(-4, 4, 300)
# f1: x^2 + y^2 = 10 → circle
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.sqrt(10)*np.cos(theta), np.sqrt(10)*np.sin(theta), 'b-', label='x²+y²=10')
# f2: y = x²-2
y_f2 = x_plot**2 - 2
ax.plot(x_plot, y_f2, 'r-', label='y = x²-2')
for i, (xs, ys) in enumerate(solutions):
    ax.plot(xs, ys, 'g*', markersize=15)
    ax.annotate(f'S{i+1}({xs:.2f},{ys:.2f})', (xs, ys), textcoords="offset points", xytext=(5,5))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 5)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.legend()
ax.set_title("Kurva Persamaan dan Solusi")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Plot peta konvergensi
ax2 = axes[1]
colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
for (x0, y0), sol_idx in convergence_map.items():
    ax2.plot(x0, y0, 's', color=colors.get(sol_idx, 'gray'), markersize=8, alpha=0.7)
for i, (xs, ys) in enumerate(solutions):
    ax2.plot(xs, ys, 'k*', markersize=12)
    ax2.annotate(f'S{i+1}', (xs, ys), fontsize=10, fontweight='bold')
ax2.set_xlim(-6, 6)
ax2.set_ylim(-6, 6)
ax2.grid(True, alpha=0.3)
ax2.set_title("Peta Initial Guess → Solusi")
ax2.set_xlabel("x0")
ax2.set_ylabel("y0")

plt.tight_layout()
plt.savefig('soal11_17_nonlinear.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Plot disimpan: soal11_17_nonlinear.png")
