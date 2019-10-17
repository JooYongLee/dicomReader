from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import numpy as np
def spsolve_lu(L, U, bin, perm_c=None, perm_r=None):
    """ an attempt to use SuperLU data to efficiently solve
        Ax = Pr.T L U Pc.T x = b
         - note that L from SuperLU is in CSC format solving for c
           results in an efficiency warning
        Pr . A . Pc = L . U
        Lc = b      - forward solve for c
         c = Ux     - then back solve for x
    """
    b = bin.copy()
    if perm_r is not None:
        b_old = b.copy()
        b[perm_r] = b_old
        # for old_ndx, new_ndx in enumerate(perm_r):
        #     print(old_ndx, new_ndx)
        #     b[new_ndx] = b_old[old_ndx]
    try:    # unit_diagonal is a new kw
        c = spsolve_triangular(L, b, lower=True, unit_diagonal=True)
    except TypeError:
        c = spsolve_triangular(L, b, lower=True)
    px = spsolve_triangular(U, c, lower=False)
    if perm_c is None:
        return px
    return px[perm_c]

A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
B = splu(A)
x = np.array([1., 2., 3.], dtype=float)

print("check", A.dot(B.solve(x)))

# print(B.perm_r)
s = spsolve_lu(B.L, B.U, x, B.perm_c, B.perm_r)
print("check with defincion", s - B.solve(x))
print("-----<>", s,  B.solve(x))
