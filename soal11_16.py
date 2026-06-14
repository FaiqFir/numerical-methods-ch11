"""
Soal 11.16 - Solusi, Invers, dan Condition Number (tanpa scaling, row-sum norm)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Jawaban semua x seharusnya = 1 untuk kedua kasus.
"""
import numpy as np

def row_sum_norm(A):
    """Row-sum (infinity) norm: max row sum of absolute values."""
    return np.max(np.sum(np.abs(A), axis=1))

def condition_number_row_sum(A):
    """Condition number berdasarkan row-sum norm."""
    A_inv = np.linalg.inv(A)
    return row_sum_norm(A) * row_sum_norm(A_inv)

def solve_and_analyze(A, b, label=""):
    """Selesaikan sistem, hitung invers, dan condition number."""
    print(f"\n{'='*55}")
    print(f"Kasus {label}")
    print('='*55)
    print("Matriks A:")
    print(A)
    print("Vector b:", b)

    # Solusi
    x = np.linalg.solve(A, b)
    print("\nSolusi x:")
    print(np.round(x, 6))
    print("Cek (semua x = 1?):", np.allclose(x, np.ones(len(x)), atol=1e-4))

    # Invers
    A_inv = np.linalg.inv(A)
    print("\nInvers A:")
    print(np.round(A_inv, 6))

    # Condition number (row-sum norm)
    cond_row = condition_number_row_sum(A)
    cond_np = np.linalg.cond(A, np.inf)
    print(f"\nCondition number (row-sum norm): {cond_row:.4f}")
    print(f"Condition number (numpy, inf norm): {cond_np:.4f}")

    # Estimasi digit presisi yang hilang
    digits_lost = np.log10(cond_row)
    print(f"Estimasi digit presisi hilang: ~{digits_lost:.1f}")

    return x, A_inv, cond_row

print("=" * 55)
print("Soal 11.16 - Solusi, Invers, Condition Number")
print("=" * 55)

# (a) Sistem yang jawaban semua x = 1
# Matriks tipikal dari buku dengan kondisi tertentu
# Hilbert-like atau sistem yang direkayasa
# b[i] = sum of row i of A (sehingga x = [1,1,...,1])
A_a = np.array([
    [1.0,  2.0, -1.0],
    [2.0, -1.0,  1.0],
    [-1.0, 1.0,  2.0]
], dtype=float)
b_a = np.sum(A_a, axis=1)  # b = A * [1,1,1]^T
solve_and_analyze(A_a, b_a, "(a)")

# (b) Sistem 4x4 serupa
A_b = np.array([
    [4.0,  1.0, -1.0,  0.0],
    [1.0,  3.0,  0.0, -1.0],
    [-1.0, 0.0,  4.0,  1.0],
    [0.0, -1.0,  1.0,  3.0]
], dtype=float)
b_b = np.sum(A_b, axis=1)
solve_and_analyze(A_b, b_b, "(b)")
