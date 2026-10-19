import numpy as np
from functools import reduce
from scipy.sparse import identity, hstack, kron, csr_matrix
from collections import deque

# Functions adapted from the ldpc package

def row_echelon(mat, reduced=False):

    m, n = np.shape(mat)
    mat = np.copy(mat)
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        if not mat[pivot_row, col]:
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            if mat[swap_row_index, col]:
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]:
            if not reduced:
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m:
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]

def rank(mat):
    return row_echelon(mat)[1]

def kernel(mat):

    transpose = mat.T
    m, _ = transpose.shape
    _, rank, transform, pivot_cols = row_echelon(transpose)
    ker = transform[rank:m]
    return ker, rank, pivot_cols

def row_basis(mat):
    return mat[row_echelon(mat.T)[3]]

def compute_code_distance(mat, is_pcm=True, is_basis=False):
    gen = mat
    if is_pcm:
        gen = kernel(mat)
    if len(gen)==0: return np.inf
    cw = gen
    if not is_basis:
        cw = row_basis(gen)
    return np.min(np.sum(cw, axis=1))

def inverse(mat):

    m, n = mat.shape
    reduced_row_ech, rank, transform, _ = row_echelon(mat, reduced=True)
    if m == n and rank == m:
        return transform
    elif m > rank and n == rank:
        return reduced_row_ech.T @ transform % 2
    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")

class css_code():
    def __init__(self, hx=np.array([[]]), hz=np.array([[]]), code_distance=np.nan, name=None, name_prefix="", check_css=False):

        self.hx = hx
        self.hz = hz

        self.lx = np.array([[]])
        self.lz = np.array([[]])

        self.N = np.nan
        self.K = np.nan
        self.D = code_distance
        self.L = np.nan
        self.Q = np.nan

        _, nx = self.hx.shape
        _, nz = self.hz.shape

        assert nx == nz, "hx and hz should have equal number of columns!"
        assert nx != 0,  "number of variable nodes should not be zero!"
        if check_css:
            assert not np.any(hx @ hz.T % 2), "CSS constraint not satisfied"

        self.N = nx
        self.hx_perp, self.rank_hx, self.pivot_hx = kernel(hx)
        self.hz_perp, self.rank_hz, self.pivot_hz = kernel(hz)
        self.hx_basis = self.hx[self.pivot_hx]
        self.hz_basis = self.hz[self.pivot_hz]
        self.K = self.N - self.rank_hx - self.rank_hz

        self.compute_ldpc_params()
        self.compute_logicals()
        if code_distance is np.nan:
            dx = compute_code_distance(self.hx_perp, is_pcm=False, is_basis=True)
            dz = compute_code_distance(self.hz_perp, is_pcm=False, is_basis=True)
            self.D = np.min([dx,dz])

        self.name = f"{name_prefix}_n{self.N}_k{self.K}" if name is None else name

    def compute_ldpc_params(self):

        hx_l = np.max(np.sum(self.hx, axis=0))
        hz_l = np.max(np.sum(self.hz, axis=0))
        self.L = np.max([hx_l, hz_l]).astype(int)

        hx_q = np.max(np.sum(self.hx, axis=1))
        hz_q = np.max(np.sum(self.hz, axis=1))
        self.Q = np.max([hx_q, hz_q]).astype(int)

    def compute_logicals(self):

        def compute_lz(ker_hx, im_hzT):
            log_stack = np.vstack([im_hzT, ker_hx])
            pivots = row_echelon(log_stack.T)[3]
            log_op_indices = [i for i in range(im_hzT.shape[0], log_stack.shape[0]) if i in pivots]
            log_ops = log_stack[log_op_indices]
            return log_ops

        self.lx = compute_lz(self.hz_perp, self.hx_basis)
        self.lz = compute_lz(self.hx_perp, self.hz_basis)

        return self.lx, self.lz

    def canonical_logicals(self):
        temp = inverse(self.lx @ self.lz.T % 2)
        self.lx = temp @ self.lx % 2

def create_circulant_matrix(l, pows):
    h = np.zeros((l,l), dtype=int)
    for i in range(l):
        for c in pows:
            h[(i+c)%l, i] = 1
    return h


def create_generalized_bicycle_codes(l, a, b, name=None):
    A = create_circulant_matrix(l, a)
    B = create_circulant_matrix(l, b)
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="GB")


def hypergraph_product(h1, h2, name=None):
    m1, n1 = np.shape(h1)
    r1 = rank(h1)
    k1 = n1 - r1
    k1t = m1 - r1

    m2, n2 = np.shape(h2)
    r2 = rank(h2)
    k2 = n2 - r2
    k2t = m2 - r2

    N = n1 * n2 + m1 * m2
    K = k1 * k2 + k1t * k2t

    h1 = csr_matrix(h1)
    hx1 = kron(h1, identity(n2, dtype=int))
    hx2 = kron(identity(m1, dtype=int), h2.T)
    hx = hstack([hx1, hx2]).toarray()

    h2 = csr_matrix(h2)
    hz1 = kron(identity(n1, dtype=int), h2)
    hz2 = kron(h1.T, identity(m2, dtype=int))
    hz = hstack([hz1, hz2]).toarray()
    return css_code(hx, hz, name=name, name_prefix="HP")

def hamming_code(rank):
    rank = int(rank)
    num_rows = (2**rank) - 1
    pcm = np.zeros((num_rows, rank), dtype=int)
    for i in range(0, num_rows):
        pcm[i] = int2bin(i+1, rank)
    return pcm.T

def rep_code(d):
    pcm = np.zeros((d-1, d), dtype=int)
    for i in range(d-1):
        pcm[i, i] = 1
        pcm[i, i+1] = 1
    return pcm

def set_pcm_row(n, pcm, row_idx, i, j):
    i1, j1 = (i+1) % n, (j+1) % n
    pcm[row_idx][i*n+j] = pcm[row_idx][i1*n+j1] = 1
    pcm[row_idx][i1*n+j] = pcm[row_idx][i*n+j1] = 1

def create_cyclic_permuting_matrix(n, shifts):
    A = np.full((n,n), -1, dtype=int)
    for i, s in enumerate(shifts):
        for j in range(n):
            A[j, (j-i)%n] = s
    return A

def create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows, name=None):
    S_l=create_circulant_matrix(l, [-1])
    S_m=create_circulant_matrix(m, [-1])
    x = kron(S_l, identity(m, dtype=int))
    y = kron(identity(l, dtype=int), S_m)
    A_list = [x**p for p in A_x_pows] + [y**p for p in A_y_pows]
    B_list = [y**p for p in B_y_pows] + [x**p for p in B_x_pows]
    A = reduce(lambda x,y: x+y, A_list).toarray()
    B = reduce(lambda x,y: x+y, B_list).toarray()
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name=name, name_prefix="BB", check_css=True), A_list, B_list

def readAlist(directory):
    alist_raw = []
    with open(directory, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(" ")
            line = list(map(int, line))
            alist_raw.append(line)
    alist_numpy = alistToNumpy(alist_raw)
    alist_numpy = alist_numpy.astype(int)
    return alist_numpy


def alistToNumpy(lines):
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=float)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix


def multiply_elements(a_b, c_d, n, m, k):
    a, b = a_b
    c, d = c_d
    return ((a + c * pow(k, b, n)) % n, (b+d) % m)

def idx2tuple(idx, m):
    b = idx % m
    a = (idx - b) / m
    return (a, b)

def create_2BGA(n, m, k, a_poly, b_poly, sr=False):
    l = n*m
    A = np.zeros((l,l))
    for (a,b) in a_poly:
        if sr:
            x = b * pow(k, a, n) % n
            b = a
            a = x
        for i in range(l):
            c, d = idx2tuple(i, m)
            a_, b_ = multiply_elements((a,b), (c,d), n, m, k)
            idx = a_ * m + b_
            A[int(idx), i] += 1

    A = A % 2

    B = np.zeros((l,l))
    for (a,b) in b_poly:
        if sr:
            x = b * pow(k, a, n) % n
            b = a
            a = x
        for i in range(l):
            c, d = idx2tuple(i, m)
            a_, b_ = multiply_elements((c,d), (a,b), n, m, k)
            idx = a_ * m + b_
            B[int(idx), i] += 1

    B = B % 2
    hx = np.hstack((A, B))
    hz = np.hstack((B.T, A.T))
    return css_code(hx, hz, name_prefix="2GBA", check_css=True)


def find_girth(pcm):
    m, n = pcm.shape
    a1 = np.hstack((np.zeros((m,m)), pcm))
    a2 = np.hstack((pcm.T, np.zeros((n,n))))
    adj_matrix = np.vstack((a1,a2))
    n = len(adj_matrix)
    girth = float('inf')

    def bfs(start):
        nonlocal girth
        distance = [-1] * n
        distance[start] = 0
        queue = deque([start])

        while queue:
            vertex = queue.popleft()
            for neighbor, is_edge in enumerate(adj_matrix[vertex]):
                if is_edge:
                    if distance[neighbor] == -1:
                        distance[neighbor] = distance[vertex] + 1
                        queue.append(neighbor)
                    elif distance[neighbor] >= distance[vertex] + 1:
                        girth = min(girth, distance[vertex] + distance[neighbor] + 1)

    for i in range(n):
        bfs(i)

    return girth if girth != float('inf') else -1

def gcd(f_coeff, g_coeff):
    return poly2coeff(gcd_inner(coeff2poly(f_coeff), coeff2poly(g_coeff)))

def gcd_inner(f, g, p=2):
    if len(f) < len(g):
        return gcd_inner(g,f,p)

    r = [0] * len(f)
    r_mult = reciprocal(g[0], p)*f[0]

    for i in range(len(f)):
        if i < len(g):
            r[i] = f[i] - g[i] * r_mult
        else:
            r[i] = f[i]
        if p != 0:
            r[i] %= p

    while abs(r[0]) < 0.0001:
        r.pop(0)
        if (len(r) == 0):
            return g

    return gcd_inner(r, g, p)

def reciprocal(n, p=0):
    if p == 0:
        return 1/n
    for i in range(p):
        if (n*i) % p == 1:
            return i
    return None

def coeff2poly(coeff):
    lead = max(coeff)
    poly = np.zeros(lead+1)
    for i in coeff:
        poly[lead-i] = 1
    return list(poly)

def poly2coeff(poly):
    l = len(poly) - 1
    return [l-i for i in range(l+1) if poly[i]][::-1]

def multiply_polynomials(a, b, m, primitive_polynomial):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & (1 << m):
            a ^= primitive_polynomial
    return result

def generate_log_antilog_tables(m, primitive_polynomial):
    gf_size = 2**m
    log_table = [-1] * gf_size
    antilog_table = [0] * gf_size

    alpha = 1
    for i in range(gf_size - 1):
        antilog_table[i] = alpha
        log_table[alpha] = i

        alpha = multiply_polynomials(alpha, 2, m, primitive_polynomial)

    log_table[0] = -1

    return log_table, antilog_table


def construct_vector(m, log_table, antilog_table):
    gf_size = 2**m
    vector = [-1] * gf_size

    for i in range(1, gf_size):
        sum_val = 1 ^ antilog_table[i % (gf_size - 1)]

        if sum_val < gf_size and log_table[sum_val] != -1:
            vector[i] = log_table[sum_val]

    return vector

def get_primitive_polynomial(m):
    if m == 2:
        primitive_polynomial = 0b111
    elif m == 3:
        primitive_polynomial = 0b1011
    elif m == 4:
        primitive_polynomial = 0b10011
    elif m == 6:
        primitive_polynomial = 0b1011011
    elif m == 8:
        primitive_polynomial = 0b100011101
    elif m == 9:
        primitive_polynomial = 0b1000010001
    elif m == 10:
        primitive_polynomial = 0b10001101111
    elif m == 12:
        primitive_polynomial = 0b1000011101011
    elif m == 15:
        primitive_polynomial = 0b1000000000110101
    else:
        raise ValueError(f"Unsupported m={m}, use the galois package to find the Conway polynomial yourself.")
    return primitive_polynomial

def create_EG_codes(s):
    order = 2 ** (2*s) - 1
    extension = 2*s
    primitive_polynomial = get_primitive_polynomial(extension)
    log_table, antilog_table = generate_log_antilog_tables(extension, primitive_polynomial)
    vector = construct_vector(extension, log_table, antilog_table)

    log_beta = 2 ** s + 1
    lines = []
    for i in range(order):
        for j in range(log_beta):
            incidence_vec = np.zeros(2 ** (2*s))
            incidence_vec[i+1] = 1

            for k in range(2 ** s):
                idx = (k * log_beta + j - i) % order
                if idx == 0:
                    incidence_vec[0] = 1
                else:
                    c = (i + vector[idx]) % order
                    incidence_vec[c+1] = 1
            lines.append(incidence_vec)

    H = np.unique(np.array(lines).astype(bool), axis=0).T
    num_row, num_col = H.shape
    assert num_col == 2 ** (2*s) + 2 ** s
    hx = np.hstack((H, np.ones((num_row,1))))
    hz = np.hstack((H, np.ones((num_row,1))))
    return  css_code(hx, hz, name_prefix=f"EG", check_css=True)
