# Tugas Metode Numerik - Bab 11: Special Matrices & Gauss-Seidel

> **Nama:** Faiq Muhammad  
> **NIM:** F5512510001  
> **Kelas:** TI A  
> **MK:** Metode Numerik

---

## Cara Menjalankan

**1. Install dependencies:**
```bash
pip install numpy scipy matplotlib
```

**2. Jalankan satu soal:**
```bash
python soal11_1.py
python soal11_2.py
python soal11_3.py
# dst, ganti nomornya sesuai soal yang ingin dijalankan
```

**3. Jalankan semua sekaligus:**

Windows (Command Prompt):
```bat
for %f in (soal11_*.py) do python %f
```

Linux / Mac:
```bash
for f in soal11_*.py; do python "$f"; done
```



---

## Library

| Library | Kegunaan |
|---------|----------|
| `numpy` | Operasi matriks, aljabar linear |
| `scipy` | Fungsi matriks lanjut, solver nonlinear |
| `matplotlib` | Visualisasi grafik |

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.1 - Thomas Algorithm untuk Sistem Tridiagonal

**File:** `soal11_1.py`

Menyelesaikan sistem tridiagonal 4×4 dari Example 11.1 menggunakan Thomas Algorithm, lalu membandingkan dengan Gauss-Seidel.

**Metode Thomas Algorithm:**
1. Forward sweep: eliminasi koefisien sub-diagonal
2. Back substitution: hitung solusi dari bawah ke atas
3. Kompleksitas O(n) vs O(n³) Gauss biasa

**Hasil:**
```
(a) Thomas Algorithm:
    x = [65.970, 93.778, 124.538, 159.480]  (max error = 0.0)

(b) Gauss-Seidel (es=5%):
    Konvergen pada iterasi 6
    x ≈ [55.85, 80.79, 114.24, 154.43]
```

---

## Soal 11.2 - Invers Matriks via LU Decomposition

**File:** `soal11_2.py`

Menghitung invers matriks A dari Example 11.1 menggunakan LU Decomposition + unit vectors.

**Metode:**
- Dekomposisi A = L·U (Doolittle)
- Untuk setiap unit vector e_i: selesaikan L·y = e_i, lalu U·x = y
- Kolom ke-i dari invers = solusi x_i

**Hasil:**
```
A⁻¹ =
[[0.755841, 0.541916, 0.349667, 0.171406],
 [0.541916, 1.105509, 0.713322, 0.349667],
 [0.349667, 0.713322, 1.105509, 0.541916],
 [0.171406, 0.349667, 0.541916, 0.755841]]

Verifikasi A · A⁻¹ = I  ✓  (max error = 5.55e-17)
```

---

## Soal 11.3 - Crank-Nicolson Tridiagonal

**File:** `soal11_3.py`

Menyelesaikan sistem tridiagonal 5×5 yang muncul dari skema Crank-Nicolson pada PDE parabolik.

**Hasil:**
```
T_1 = 59.737453
T_2 = 81.022587
T_3 = 104.691909
T_4 = 131.675624
T_5 = 163.034190
```

---

## Soal 11.4 - Verifikasi Cholesky Decomposition

**File:** `soal11_4.py`

Memverifikasi hasil Cholesky dari Example 11.2 dengan menghitung L · Lᵀ dan membandingkan dengan A.

**Matriks A (Example 11.2):**
```
A = [[6, 15, 55], [15, 55, 225], [55, 225, 979]]
```

**Hasil dekomposisi:**
```
L = [[2.449490,  0.000000,  0.000000],
     [6.123724,  4.183300,  0.000000],
     [22.453656, 20.916501, 6.110101]]

Verifikasi L · Lᵀ = A  ✓  (max error = 8.88e-16)
```

---

## Soal 11.5 - Cholesky untuk Sistem Simetris

**File:** `soal11_5.py`

Melakukan Cholesky decomposition pada sistem simetris 3×3, lalu menyelesaikan A·x = b.

**Langkah:**
1. A = L·Lᵀ (Cholesky)
2. L·y = b (forward substitution)
3. Lᵀ·x = y (back substitution)

**Hasil:** `x = [1.0, 1.0, 1.0]`  (max error = 2.22e-16) ✓

---

## Soal 11.6 - Cholesky Decomposition Manual

**File:** `soal11_6.py`

Melakukan Cholesky step-by-step pada matriks 3×3 dengan menampilkan setiap langkah perhitungan.

**Matriks A:**
```
A = [[1, 1, 1], [1, 5, 5], [1, 5, 14]]
```

**Hasil L:**
```
L = [[1, 0, 0],
     [1, 2, 0],
     [1, 2, 3]]

Verifikasi L · Lᵀ = A  ✓  (max error = 0.0)
```

---

## Soal 11.7 - Cholesky + Verifikasi Persamaan (11.3) & (11.4)

**File:** `soal11_7.py`

Menerapkan Cholesky pada matriks 2×2 dan memverifikasi bahwa hasilnya sesuai dengan persamaan buku.

**Persamaan kunci:**
- Eq. (11.3): `l_jj = sqrt(a_jj − Σ l_jk²)`
- Eq. (11.4): `l_ij = (a_ij − Σ l_ik·l_jk) / l_jj`  untuk i > j

**Hasil:**
```
A = [[4, 2], [2, 3]]
L = [[2.0, 0.0], [1.0, 1.414]]

l_11 = sqrt(4) = 2.0           ← Eq. (11.3)
l_21 = 2/2 = 1.0               ← Eq. (11.4)
l_22 = sqrt(3 − 1²) = 1.414   ← Eq. (11.3)

Rekonstruksi L · Lᵀ = A  ✓  (max error = 4.44e-16)
```

---

## Soal 11.8 - Gauss-Seidel + Overrelaxation (λ=1.2)

**File:** `soal11_8.py`

Menyelesaikan sistem tridiagonal dari Prob. 11.1 dengan Gauss-Seidel biasa dan dengan SOR λ=1.2.

**Formula SOR:** `x_i^(k+1) = λ · x_i^GS + (1−λ) · x_i^(k)`

**Hasil (es = 5%):**
```
(a) λ=1.0: Konvergen iterasi 8,  x ≈ [61.97, 88.64, 120.46, 157.48]
(b) λ=1.2: Konvergen iterasi 6,  x ≈ [63.90, 91.69, 123.20, 158.95]
    Solusi eksak:                 x  = [65.97, 93.78, 124.54, 159.48]
```
Overrelaxation λ=1.2 lebih cepat 2 iterasi.

---

## Soal 11.9 - Gauss-Seidel: Reaktor Tersambung

**File:** `soal11_9.py`

Menyelesaikan sistem 3 reaktor tersambung (dari Prob. 10.8) dengan Gauss-Seidel.

**Sistem:**
```
15c1 -  3c2 -  c3 = 3800
-3c1 + 18c2 - 6c3 = 1200
-4c1 -   c2 +12c3 = 2350
```

**Hasil (es = 5%):**
```
Konvergen pada iterasi 4
c1 = 319.33 g/m³,  c2 = 226.54 g/m³,  c3 = 321.15 g/m³
(eksak: c1=320.21, c2=227.20, c3=321.50)
```

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.10 - Jacobi Iteration

**File:** `soal11_10.py`

Menyelesaikan sistem yang sama dengan Prob. 11.9 menggunakan Jacobi iteration dan membandingkan hasilnya.

**Perbedaan Jacobi vs Gauss-Seidel:**
- Jacobi: semua x diperbarui menggunakan nilai lama x^(k)
- Gauss-Seidel: nilai baru langsung digunakan pada iterasi yang sama

**Hasil (es = 5%):**
```
Jacobi:       c1=315.29, c2=219.07, c3=315.62  (4 iterasi)
Gauss-Seidel: c1=319.33, c2=226.54, c3=321.15  (4 iterasi)
Eksak:        c1=320.21, c2=227.20, c3=321.50
```
Gauss-Seidel lebih akurat dengan jumlah iterasi yang sama.

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.11 - Gauss-Seidel hingga es < 5%

**File:** `soal11_11.py`

**Sistem:**
```
10x1 - 2x2 -  x3 =  27.0
-2x1 +10x2 -  x3 = -61.5
 -x1 -  x2 + 5x3 = -21.5
```

**Hasil:**
```
Konvergen pada iterasi 4 (max error = 2.82% < 5%)
x1 = 0.8552,  x2 = -6.5216,  x3 = -5.4333
(eksak: x1=0.852, x2=-6.523, x3=-5.434)
```

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.12 - Gauss-Seidel + Underrelaxation (λ=0.95)

**File:** `soal11_12.py`

Sistem awalnya tidak diagonally dominant sehingga baris harus ditukar sebelum iterasi.

**Setelah swap baris:**
```
 6x1 -  x2 = 15
-2x1 + 12x2 = 40
```

**Hasil (es = 5%):**
```
(a) λ=1.0: Konvergen iterasi 3,  x1=3.14236, x2=3.85706
(b) λ=0.95: Konvergen iterasi 3, x1=3.13375, x2=3.85422
    Solusi eksak: x1=3.14286, x2=3.85714
```

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.13 - Gauss-Seidel + Overrelaxation (λ=1.2)

**File:** `soal11_13.py`

**Sistem (diagonally dominant):**
```
 8x1 - 3x2 + 2x3 = 20
 4x1 +11x2 -  x3 = 33
 6x1 + 3x2 +12x3 = 36
```

**Hasil (es = 5%):**
```
(a) λ=1.0: Konvergen iterasi 3,  x1=3.010, x2=1.997, x3=0.996
(b) λ=1.2: Konvergen iterasi 4,  x1=3.014, x2=1.991, x3=0.989
    Solusi eksak: x1=3.0, x2=2.0, x3=1.0
```
Untuk sistem ini, λ=1.0 lebih efisien dari overrelaxation.

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.14 - Gambar Ulang Fig. 11.5 (slope +1 dan −1)

**File:** `soal11_14.py`  
**Output plot:** `soal11_14_fig115.png`

**Sistem:**
```
x1 + x2 = 3   (slope = −1)
x1 − x2 = 1   (slope = +1)
Solusi eksak: x1=2, x2=1
```

**Cek dominansi diagonal:**
- Baris 1: |1| vs |1| → **tidak dominan**
- Baris 2: |1| vs |1| → **tidak dominan**

**Hasil iterasi Gauss-Seidel (dari x=[0,0]):**
```
Iter 1: x1=3.0, x2=2.0
Iter 2: x1=1.0, x2=0.0
Iter 3: x1=3.0, x2=2.0
Iter 4: x1=1.0, x2=0.0  ← osilasi terus, tidak konvergen
```

**Kesimpulan:** Ketika slope kedua persamaan adalah +1 dan −1, matriks tidak diagonally dominant (|a_ii| = Σ|a_ij|). Gauss-Seidel berosilasi permanen di antara dua titik dan **tidak pernah konvergen** ke solusi eksak (2, 1). Plot menggambarkan jalur zig-zag yang berulang tanpa mendekati titik perpotongan.

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.15 - Identifikasi Set yang Tidak Konvergen

**File:** `soal11_15.py`

Memeriksa tiga set persamaan linear untuk menentukan apakah Gauss-Seidel akan konvergen.

**Kriteria:** Syarat cukup konvergensi = dominansi diagonal ketat: `|a_ii| > Σ|a_ij|` (j≠i)

**Hasil:**
```
Set One:   TIDAK Konvergen ✗  (nilai meledak / divergen)
Set Two:   TIDAK Konvergen ✗
Set Three: TIDAK Konvergen ✗
```
Ketiga set tidak memenuhi syarat dominansi diagonal sehingga Gauss-Seidel tidak konvergen.

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.16 - Solusi, Invers, dan Condition Number

**File:** `soal11_16.py`

Untuk dua sistem (a) dan (b) dengan solusi eksak semua x = 1:
1. Hitung solusi `Ax = b`
2. Hitung invers `A⁻¹`
3. Hitung condition number berdasarkan **row-sum norm**: `κ(A) = ‖A‖∞ · ‖A⁻¹‖∞`

**Digits presisi hilang ≈ log₁₀(κ)**

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.17 - Sistem Nonlinear Simultan

**File:** `soal11_17.py`  
**Output plot:** `soal11_17_nonlinear.png`

**Sistem:**
```
f1: x² + y² = 10
f2: y = x² − 2
```

**Hasil (a) - dua pasang solusi:**
```
Solusi 1: x =  2.091,  y = 2.372
Solusi 2: x = -2.091,  y = 2.372
```

**(b)** Plot peta initial guess menunjukkan basin of attraction setiap solusi untuk x₀, y₀ ∈ [−6, 6].

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.18 - Produksi Elektronik

**File:** `soal11_18.py`

**Setup sistem** (T=transistor, R=resistor, C=chip):
```
4T + 3R + 2C = 960  (tembaga)
 T + 3R +  C = 510  (seng)
2T +  R + 3C = 610  (kaca)
```

**Hasil produksi minggu ini:**
```
Transistor = 120 unit
Resistor   = 100 unit
Chip       =  90 unit
```
Verifikasi: semua material terpakai tepat sesuai ketersediaan ✓

---

## File yang Dihasilkan Setelah Run

| File | Keterangan |
|------|------------|
| `soal11_14_fig115.png` | Grafik jalur iterasi Gauss-Seidel yang berosilasi (slope +1 dan -1) |
| `soal11_17_nonlinear.png` | Grafik kurva sistem nonlinear dan peta basin of attraction |
| `soal11_19_hilbert.png` | Grafik error per komponen dan condition number Hilbert matrix |
| `soal11_20_vandermonde.png` | Grafik perbandingan condition number Vandermonde vs Hilbert |
| `soal11_23_operations.png` | Grafik perbandingan jumlah operasi Thomas Algorithm vs Gauss Elimination |
| `soal11_27_concentration.png` | Grafik profil konsentrasi solusi numerik vs analitik |

---

## Soal 11.19 - Hilbert Matrix 10 Dimensi

**File:** `soal11_19.py`  
**Output plot:** `soal11_19_hilbert.png`

`H[i,j] = 1/(i+j+1)` - contoh klasik matriks sangat ill-conditioned.

**Hasil:**
```
Spectral condition number: 1.602e+13
log₁₀(κ) = 13.20
Digit presisi hilang: ~13.2 dari ~15.6 total (float64)
Digit yang masih dapat dipercaya: ~2.4
```
Untuk n=10, hampir semua presisi numerik habis karena ill-conditioning.

---

## Soal 11.20 - Vandermonde Matrix 6 Dimensi

**File:** `soal11_20.py`  
**Output plot:** `soal11_20_vandermonde.png`

`x = [4, 2, 7, 10, 3, 5]`,  `V[i,j] = xᵢ^(n−1−j)`

**Hasil:**
```
Spectral condition number: 1.449e+07
Max error solusi: 1.72e-11
```
Vandermonde 6×6 lebih baik dari Hilbert 10×10 tetapi tetap ill-conditioned.

---

## Soal 11.21 - Augmentasi Matriks [A | I]

> **Soal ini bersifat deskriptif - hanya minta satu baris perintah MATLAB.**

**Pertanyaan:** Tulis satu baris perintah MATLAB yang membuat matriks `Aug = [A | I]`.

**Jawaban (MATLAB):**
```matlab
Aug = [A, eye(size(A, 1))]
```

**Penjelasan:**
- `size(A, 1)` mengambil jumlah baris A
- `eye(n)` membuat matriks identitas n×n
- `[A, eye(...)]` menggabungkan A dan I secara horizontal
- Hasilnya adalah matriks n × 2n berbentuk `[A | I]`

**Ekuivalen Python/NumPy:**
```python
Aug = np.hstack([A, np.eye(len(A))])
```

Matriks augmentasi ini berguna untuk menghitung A⁻¹ via eliminasi Gauss-Jordan: setelah reduksi baris penuh, bagian kanan menjadi A⁻¹.

---

## Soal 11.22 - Bentuk Matriks, Transpose, Invers

**File:** `soal11_22.py`

**Sistem:**
```
 4x1 + 2x2 -  x3 = -3
-2x1 + 5x2 -  x3 = -7
  x1 - 2x2 + 6x3 =  6
```

**Hasil:**
- Solusi: `x1 = -0.5, x2 = -1.0, x3 = 0.5`
- Transpose Aᵀ: baris dan kolom dipertukar
- Invers A⁻¹: dihitung via `np.linalg.inv`
- Verifikasi A · A⁻¹ = I ✓

---

## Soal 11.23 - Analisis Operasi: Thomas vs Gauss Elimination

**File:** `soal11_23.py`  
**Output plot:** `soal11_23_operations.png`

| Metode | Formula Operasi |
|--------|----------------|
| Thomas Algorithm | ~5n − 4 |
| Gauss Elimination | ~(2n³ + 3n² − 5n)/3 |

**Perbandingan untuk berbagai n:**
```
n= 5:  Thomas=21,    Gauss=60     (Gauss 2.9× lebih banyak)
n=10:  Thomas=46,    Gauss=430    (Gauss 9.3× lebih banyak)
n=20:  Thomas=96,    Gauss=3127   (Gauss 32.6× lebih banyak)
```
Keunggulan Thomas makin besar seiring n meningkat karena kompleksitas O(n) vs O(n³).

---

## Soal 11.24 - Program Thomas Algorithm (User-Friendly)

**File:** `soal11_24.py`

Program lengkap dengan tampilan langkah-langkah forward sweep dan back substitution.

**Test: Menduplikat Example 11.1**
```
Input: b=[2.04,2.04,2.04,2.04], c=a=[-1,-1,-1], d=[40.8,0.8,0.8,200.8]
Hasil: x=[65.970, 93.778, 124.538, 159.480]
Max error vs numpy: 0.00e+00  ✓
```

---

## Soal 11.25 - Program Cholesky Decomposition (User-Friendly)

**File:** `soal11_25.py`

Program lengkap dengan tampilan setiap langkah dekomposisi, forward dan back substitution.

**Test: Menduplikat Example 11.2**
```
A = [[6,15,55],[15,55,225],[55,225,979]],  b = [76, 295, 1259]

L = [[2.449,  0.000,  0.000],
     [6.124,  4.183,  0.000],
     [22.454, 20.917,  6.110]]

Max error: 2.27e-13  ✓
```

---

## Soal 11.26 - Program Gauss-Seidel (User-Friendly)

**File:** `soal11_26.py`

Program lengkap dengan tabel iterasi, opsi SOR (omega), dan cek dominansi diagonal.

**Test: Menduplikat Example 11.3**
```
Sistem: 3x1-0.1x2-0.2x3=7.85, 0.1x1+7x2-0.3x3=-19.3, 0.3x1-0.2x2+10x3=71.4

Konvergen pada iterasi 3 (max error = 0.32% < 5%)
x1 ≈ 3.0,  x2 ≈ -2.5,  x3 ≈ 7.0  ✓
```

---

## Soal 11.27 - PDE 1D: Diffusi-Konveksi-Reaksi

**File:** `soal11_27.py`  
**Output plot:** `soal11_27_concentration.png`

**Persamaan:** `D·∂²c/∂x² − U·∂c/∂x − k·c = 0`  
**Parameter:** D=2, U=1, k=0.2, c(0)=80, c(10)=20, Δx=2

**Diskretisasi finite difference:**
```
a·c_{i-1} + b·c_i + c·c_{i+1} = 0
  a = D/Δx² + U/(2Δx) =  0.75
  b = -2D/Δx² - k      = -1.20
  c = D/Δx² - U/(2Δx) =  0.25
```

**Profil konsentrasi:**
```
c(0) = 80.0  (BC)
c(2) ≈ 67.0
c(4) ≈ 55.6
c(6) ≈ 44.1
c(8) ≈ 31.5
c(10)= 20.0  (BC)
```
Plot dibandingkan dengan solusi analitik.

---

## Soal 11.28 - Pentadiagonal Solver

**File:** `soal11_28.py`

Menyelesaikan sistem pentadiagonal (bandwidth 5) secara efisien.

**Struktur matriks:**
```
[b  c  d  .  .  .]
[a  b  c  d  .  .]
[e  a  b  c  d  .]
[.  e  a  b  c  d]
[.  .  e  a  b  c]
[.  .  .  e  a  b]
```
- b = main diagonal
- c, d = super-diagonal (+1, +2)
- a, e = sub-diagonal  (−1, −2)

**Test sistem 6×6** (b=8, a=c=−1, d=e=−1, RHS=10):
```
x = [1.788, 2.040, 2.267, 2.267, 2.040, 1.788]
Max error vs numpy: 0.00e+00  ✓
```

**Test sistem 10×10:** Max error = 0.00e+00 ✓

Algoritma ini mirip Thomas Algorithm tetapi menangani dua diagonal di setiap sisi (bandwidth 5), dengan kompleksitas O(n).

---

## Ringkasan Metode

| Soal | Metode | Kompleksitas |
|------|--------|-------------|
| 11.1, 11.3, 11.24 | Thomas Algorithm | O(n) |
| 11.2 | LU Decomposition + Unit Vectors | O(n³) |
| 11.4–11.7, 11.25 | Cholesky Decomposition | O(n³/3) |
| 11.8–11.13, 11.26 | Gauss-Seidel / SOR | O(kn²) |
| 11.10 | Jacobi Iteration | O(kn²) |
| 11.16, 11.19–11.20 | Condition Number Analysis | - |
| 11.17 | Nonlinear System (scipy.fsolve) | - |
| 11.23 | Operation Count Analysis | - |
| 11.27 | Finite Difference (PDE) | O(n) tridiagonal |
| 11.28 | Pentadiagonal Solver | O(n) |

### Syarat Konvergensi Gauss-Seidel
Syarat **cukup** (bukan perlu): dominansi diagonal ketat
```
|a_ii| > Σ|a_ij|  untuk semua i, j≠i
```
Jika tidak terpenuhi: tukar urutan baris, atau gunakan underrelaxation (λ<1).
