"""
Soal 11.25 - Program Cholesky Decomposition (User-Friendly)
Sumber: Numerical Methods for Engineers, Chapra & Canale

Ditest menggunakan hasil Example 11.2.
"""
import numpy as np

def cholesky_decomposition(A, verbose=True):
    """
    Cholesky Decomposition: A = L * L^T
    Hanya untuk matriks simetris positif definit.
    
    Parameters:
    -----------
    A       : matriks simetris positif definit (n x n)
    verbose : tampilkan langkah-langkah
    
    Returns:
    --------
    L : matriks lower triangular
    """
    n = len(A)
    A = np.array(A, dtype=float)
    L = np.zeros((n, n))
    
    # Cek simetri
    if not np.allclose(A, A.T):
        raise ValueError("Matriks harus simetris!")
    
    if verbose:
        print("\n--- Langkah Cholesky Decomposition ---")
    
    for j in range(n):
        # Diagonal element (Eq. 11.3)
        s = sum(L[j][k]**2 for k in range(j))
        val = A[j][j] - s
        if val <= 0:
            raise ValueError(f"Matriks tidak positif definit (nilai negatif pada l_{j+1}{j+1})")
        L[j][j] = np.sqrt(val)
        
        if verbose:
            print(f"  l_{j+1}{j+1} = sqrt({A[j][j]:.6f} - {s:.6f}) = {L[j][j]:.6f}")
        
        # Off-diagonal elements (Eq. 11.4)
        for i in range(j+1, n):
            s2 = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / L[j][j]
            if verbose:
                print(f"  l_{i+1}{j+1} = ({A[i][j]:.6f} - {s2:.6f}) / {L[j][j]:.6f} = {L[i][j]:.6f}")
    
    return L

def cholesky_solve(L, b, verbose=True):
    """
    Solusi A*x = b menggunakan hasil Cholesky.
    L*L^T*x = b → L*y = b → L^T*x = y
    """
    n = len(b)
    b = np.array(b, dtype=float)
    
    # Forward substitution: L*y = b
    y = np.zeros(n)
    if verbose:
        print("\n--- Forward Substitution: L*y = b ---")
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
        if verbose:
            print(f"  y[{i+1}] = ({b[i]:.6f} - {s:.6f}) / {L[i][i]:.6f} = {y[i]:.6f}")
    
    # Back substitution: L^T*x = y
    x = np.zeros(n)
    LT = L.T
    if verbose:
        print("\n--- Back Substitution: L^T*x = y ---")
    for i in range(n-1, -1, -1):
        s = sum(LT[i][j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - s) / LT[i][i]
        if verbose:
            print(f"  x[{i+1}] = ({y[i]:.6f} - {s:.6f}) / {LT[i][i]:.6f} = {x[i]:.6f}")
    
    return x

# ==============================================
# TEST: Duplikat Example 11.2
# ==============================================
print("=" * 55)
print("Soal 11.25 - Program Cholesky Decomposition")
print("=" * 55)
print("\nTest: Menduplikat hasil Example 11.2")

# Matriks dari Example 11.2
A = np.array([
    [6.0,  15.0,  55.0],
    [15.0, 55.0,  225.0],
    [55.0, 225.0, 979.0]
], dtype=float)

b = np.array([76.0, 295.0, 1259.0], dtype=float)

print("\nMatriks A:")
print(A)
print("Vector b:", b)

# Dekomposisi Cholesky
L = cholesky_decomposition(A, verbose=True)

print("\nMatriks L:")
print(np.round(L, 6))

print("\nMatriks L^T:")
print(np.round(L.T, 6))

# Verifikasi dekomposisi
print("\nVerifikasi L * L^T = A:")
print(np.round(L @ L.T, 6))

# Solusi
x = cholesky_solve(L, b, verbose=True)
print("\n" + "=" * 40)
print("SOLUSI:")
for i, xi in enumerate(x):
    print(f"  x[{i+1}] = {xi:.6f}")

# Verifikasi
print("\nVerifikasi A*x = b:")
print("A*x =", np.round(A @ x, 6))
print("b   =", b)
print(f"Max error: {np.max(np.abs(A@x - b)):.2e}")
print("\n✓ Hasil sesuai dengan Example 11.2")
