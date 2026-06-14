"""
Soal 11.27 - PDE 1D: Diffusi-Konveksi-Reaksi (Finite Difference)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Persamaan diferensial:
  D * d²c/dx² - U * dc/dx - k*c = 0

Parameter:
  D = 2, U = 1, k = 0.2
  c(0) = 80, c(10) = 20
  Δx = 2 (dari x=0 ke x=10)
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameter
D = 2.0
U = 1.0
k = 0.2
c0 = 80.0   # c(0)
cL = 20.0   # c(10)
L = 10.0    # panjang domain
dx = 2.0    # langkah spasial

print("=" * 60)
print("Soal 11.27 - PDE 1D: Diffusi-Konveksi-Reaksi")
print("=" * 60)
print(f"\nPersamaan: D*d²c/dx² - U*dc/dx - k*c = 0")
print(f"D = {D}, U = {U}, k = {k}")
print(f"c(0) = {c0}, c(10) = {cL}")
print(f"Δx = {dx}")

# Buat grid titik interior
x_nodes = np.arange(0, L + dx, dx)
n_total = len(x_nodes)
print(f"\nTitik grid: {x_nodes}")
print(f"Total titik: {n_total} (termasuk 2 boundary)")

# Titik interior: x = 2, 4, 6, 8
x_interior = x_nodes[1:-1]
n_int = len(x_interior)
print(f"Titik interior: {x_interior} ({n_int} titik)")

# Diskretisasi finite difference:
# D*(c_{i-1} - 2c_i + c_{i+1})/dx² - U*(c_{i+1} - c_{i-1})/(2dx) - k*c_i = 0
# 
# Koefisien:
# a_i (untuk c_{i-1}): D/dx² + U/(2dx)
# b_i (untuk c_i):     -2D/dx² - k
# c_i (untuk c_{i+1}): D/dx² - U/(2dx)

a_coef = D/dx**2 + U/(2*dx)   # lower diagonal
b_coef = -2*D/dx**2 - k        # main diagonal
c_coef = D/dx**2 - U/(2*dx)   # upper diagonal

print(f"\nKoefisien finite difference:")
print(f"  a (lower) = D/dx² + U/(2dx) = {a_coef:.4f}")
print(f"  b (main)  = -2D/dx² - k     = {b_coef:.4f}")
print(f"  c (upper) = D/dx² - U/(2dx) = {c_coef:.4f}")

# Bangun matriks sistem
A = np.zeros((n_int, n_int))
rhs = np.zeros(n_int)

for i in range(n_int):
    A[i][i] = b_coef
    if i > 0:
        A[i][i-1] = a_coef
    if i < n_int - 1:
        A[i][i+1] = c_coef

# Kondisi batas (pindahkan ke RHS)
rhs[0] = -a_coef * c0    # dari titik x=0
rhs[-1] = -c_coef * cL   # dari titik x=10

print(f"\nMatriks A ({n_int}x{n_int}):")
print(np.round(A, 4))
print(f"\nVector RHS: {np.round(rhs, 4)}")

# Solusi
c_interior = np.linalg.solve(A, rhs)

# Profil lengkap konsentrasi
c_full = np.zeros(n_total)
c_full[0] = c0
c_full[-1] = cL
c_full[1:-1] = c_interior

print("\nSolusi konsentrasi:")
for xi, ci in zip(x_nodes, c_full):
    print(f"  c({xi:.0f}) = {ci:.4f} g/m³")

# Solusi analitik (untuk verifikasi)
# D*c'' - U*c' - k*c = 0
# Karakteristik: D*r² - U*r - k = 0
r = np.roots([D, -U, -k])
print(f"\nAkar karakteristik: r1={r[0]:.4f}, r2={r[1]:.4f}")

# c(x) = A1*exp(r1*x) + A2*exp(r2*x)
# Kondisi batas: c(0)=80, c(10)=20
M = np.array([[1, 1],
              [np.exp(r[0]*L), np.exp(r[1]*L)]])
constants = np.linalg.solve(M, [c0, cL])
A1, A2 = constants

x_analytical = np.linspace(0, L, 200)
c_analytical = A1 * np.exp(r[0] * x_analytical) + A2 * np.exp(r[1] * x_analytical)

# Plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(x_analytical, c_analytical, 'b-', linewidth=2, label='Solusi Analitik')
ax.plot(x_nodes, c_full, 'r-o', markersize=10, linewidth=1.5, label=f'Finite Difference (Δx={dx})')
ax.set_xlabel('Jarak x (m)', fontsize=12)
ax.set_ylabel('Konsentrasi c (g/m³)', fontsize=12)
ax.set_title('Profil Konsentrasi: PDE Diffusi-Konveksi-Reaksi\nD=2, U=1, k=0.2, c(0)=80, c(10)=20', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('soal11_27_concentration.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n✓ Plot disimpan: soal11_27_concentration.png")
