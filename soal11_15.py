"""
Soal 11.15 - Identifikasi Set Persamaan yang TIDAK Konvergen dengan Gauss-Seidel
Sumber: Numerical Methods for Engineers, Chapra & Canale

Set One:   8x - 13y + z = 12
            2x - 6x + 17z = 1  (perhatian: ada typo di soal, ini 2x - 6? + 17z)
                14y - z = 5

Set Two:   x + y + 5z = 7
            x + 4y - z = 4
           3x + y - z = 4

Set Three: 2x + 3y + 5z = 7
           -2x + 4y - 5z = -3
                2y - z = 1
"""
import numpy as np

def check_diagonal_dominance(A, label=""):
    """Cek dominansi diagonal."""
    n = len(A)
    print(f"\nDominasi Diagonal - {label}:")
    dominant = True
    for i in range(n):
        d = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        status = "✓" if d >= off else "✗ TIDAK dominan"
        print(f"  Baris {i+1}: |a_ii|={d:.1f} vs sum|a_ij|={off:.1f} {status}")
        if d < off:
            dominant = False
    return dominant

def gauss_seidel_test(A, b, label="", max_iter=25):
    """Tes Gauss-Seidel dan tampilkan apakah konvergen."""
    n = len(b)
    x = np.zeros(n)
    print(f"\nIterasi Gauss-Seidel - {label} (max 25 iter):")
    print(f"{'Iter':>5} | " + " ".join([f"{'x'+str(i+1):>12}" for i in range(n)]) + " | Status")
    print("-" * 65)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        # Relative error
        ea = [abs((x[i]-x_old[i])/x[i])*100 if abs(x[i])>1e-10 else float('inf') for i in range(n)]
        max_ea = max(ea)

        # Cek divergensi
        if np.any(np.abs(x) > 1e8):
            print(f"{k+1:>5} | " + " ".join([f"{xi:>12.2e}" for xi in x]) + " | DIVERGEN!")
            return x, False

        print(f"{k+1:>5} | " + " ".join([f"{xi:>12.6f}" for xi in x]) + f" | ea={max_ea:.2f}%")

        if max_ea < 1.0:
            print(f"  → Konvergen pada iterasi {k+1}")
            return x, True

    print(f"  → Tidak konvergen dalam {max_iter} iterasi")
    return x, False

print("=" * 65)
print("Soal 11.15 - Identifikasi Set yang Tidak Konvergen")
print("=" * 65)

# SET ONE
A1 = np.array([
    [8.0, -13.0,  1.0],
    [2.0,  -6.0, 17.0],
    [0.0,  14.0, -1.0]
], dtype=float)
b1 = np.array([12.0, 1.0, 5.0], dtype=float)

print("\n========== SET ONE ==========")
print("8x - 13y + z = 12")
print("2x - 6y + 17z = 1")
print("14y - z = 5")
d1 = check_diagonal_dominance(A1, "Set One")
x1, conv1 = gauss_seidel_test(A1, b1, "Set One")

# SET TWO
A2 = np.array([
    [1.0,  1.0,  5.0],
    [1.0,  4.0, -1.0],
    [3.0,  1.0, -1.0]
], dtype=float)
b2 = np.array([7.0, 4.0, 4.0], dtype=float)

print("\n========== SET TWO ==========")
print("x + y + 5z = 7")
print("x + 4y - z = 4")
print("3x + y - z = 4")
d2 = check_diagonal_dominance(A2, "Set Two")
x2, conv2 = gauss_seidel_test(A2, b2, "Set Two")

# SET THREE
A3 = np.array([
    [2.0,  3.0,  5.0],
    [-2.0, 4.0, -5.0],
    [0.0,  2.0, -1.0]
], dtype=float)
b3 = np.array([7.0, -3.0, 1.0], dtype=float)

print("\n========== SET THREE ==========")
print("2x + 3y + 5z = 7")
print("-2x + 4y - 5z = -3")
print("2y - z = 1")
d3 = check_diagonal_dominance(A3, "Set Three")
x3, conv3 = gauss_seidel_test(A3, b3, "Set Three")

print("\n" + "=" * 65)
print("RINGKASAN:")
print(f"  Set One:   {'Konvergen ✓' if conv1 else 'TIDAK Konvergen ✗'}")
print(f"  Set Two:   {'Konvergen ✓' if conv2 else 'TIDAK Konvergen ✗'}")
print(f"  Set Three: {'Konvergen ✓' if conv3 else 'TIDAK Konvergen ✗'}")
print("\nKriteria: Sistem tidak konvergen jika |a_ii| < Σ|a_ij| (j≠i)")
print("atau jika nilai iterasi melebihi batas yang wajar.")
