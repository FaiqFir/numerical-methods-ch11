"""
Soal 11.18 - Produksi Elektronik: Transistor, Resistor, Chip
Sumber: Numerical Methods for Engineers, Chapra & Canale

| Komponen     | Tembaga | Seng | Kaca |
|--------------|---------|------|------|
| Transistor   |    4    |  1   |  2   |
| Resistor     |    3    |  3   |  1   |
| Chip         |    2    |  1   |  3   |

Material tersedia: 960 tembaga, 510 seng, 610 kaca.
"""
import numpy as np

print("=" * 60)
print("Soal 11.18 - Sistem Produksi Elektronik")
print("=" * 60)
print("\nTabel kebutuhan material:")
print("  Komponen   | Tembaga | Seng | Kaca")
print("  -----------|---------|------|------")
print("  Transistor |    4    |  1   |  2  ")
print("  Resistor   |    3    |  3   |  1  ")
print("  Chip       |    2    |  1   |  3  ")
print("\nMaterial tersedia: 960 tembaga, 510 seng, 610 kaca")

# Misalkan T = transistor, R = resistor, C = chip
# 4T + 3R + 2C = 960  (tembaga)
# 1T + 3R + 1C = 510  (seng)
# 2T + 1R + 3C = 610  (kaca)

A = np.array([
    [4.0, 3.0, 2.0],
    [1.0, 3.0, 1.0],
    [2.0, 1.0, 3.0]
], dtype=float)

b = np.array([960.0, 510.0, 610.0], dtype=float)

print("\nMatriks koefisien A:")
print(A)
print("Vector b (sumber daya):", b)

# Solusi dengan numpy
x = np.linalg.solve(A, b)
T, R, C = x

print("\n" + "=" * 40)
print("HASIL PRODUKSI MINGGU INI:")
print("=" * 40)
print(f"  Transistor (T) = {T:.1f} unit")
print(f"  Resistor   (R) = {R:.1f} unit")
print(f"  Chip       (C) = {C:.1f} unit")

# Verifikasi
print("\nVerifikasi penggunaan material:")
print(f"  Tembaga: 4*{T:.0f} + 3*{R:.0f} + 2*{C:.0f} = {4*T + 3*R + 2*C:.0f} (tersedia: 960)")
print(f"  Seng:    1*{T:.0f} + 3*{R:.0f} + 1*{C:.0f} = {1*T + 3*R + 1*C:.0f} (tersedia: 510)")
print(f"  Kaca:    2*{T:.0f} + 1*{R:.0f} + 3*{C:.0f} = {2*T + 1*R + 3*C:.0f} (tersedia: 610)")

residual = A @ x - b
print(f"\nMax residual: {np.max(np.abs(residual)):.2e}")

# Invers matriks
A_inv = np.linalg.inv(A)
print("\nInvers matriks A (berguna untuk variasi material mingguan):")
print(np.round(A_inv, 6))

# Condition number
cond = np.linalg.cond(A)
print(f"\nCondition number: {cond:.4f} (sistem {'well-conditioned' if cond < 100 else 'ill-conditioned'})")

print("\n" + "="*40)
print("Kesimpulan:")
print(f"  Produksi: {int(round(T))} transistor, {int(round(R))} resistor, {int(round(C))} chip")
