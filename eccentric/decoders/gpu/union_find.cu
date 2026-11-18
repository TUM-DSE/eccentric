// Union-find decoding with inversion on the GPU, one shot per thread block: the algorithm of ldpc 2.4.1's
// UfDecoder::matrix_decode (src_cpp/union_find.hpp, MIT licence, Joschka Roffe) with bits_per_step = number of
// bits (as ECCentric's union_find calls it) and fixed bit weights.
//
//   Each detection event starts a cluster. While some active cluster is invalid, the invalid ones, smallest
//   first, each grow by the bits next to their checks (all of them, unless the list of (check, bit) pairs is
//   longer than the number of bits: then those of its first n pairs by weight), merging with every cluster they touch, and are
//   then solved: the cluster's checks against its bits ordered by weight, the solution on the first linearly
//   independent bits (Gaussian elimination in that order, free bits 0). A cluster is valid once its checks'
//   syndrome is in the span of its bits. The decoding is the union of the active clusters' solutions.
//
// ldpc breaks ties (bits of equal weight, clusters of equal size) by the iteration order of its hash sets; here
// bits go by the order of `perm` (union_find_gpu.py: a fixed random order among equal weights) and clusters by
// their first detection event. The logical error rate depends on the tie order, so it differs from ldpc's by
// 0.1-2% (relative), a known difference (analysis/GPU.md). Clusters are merged into the growing one through a
// disjoint-set forest over the detection events.

__device__ __forceinline__ int find_root(const int* parent, int x) {
    while (true) {
        const int p = ((volatile const int*)parent)[x];
        if (p == x) return x;
        x = p;
    }
}

// Exclusive prefix sum over the block of one int per thread; returns this thread's offset, *total the sum
__device__ int block_scan(int value, int* scratch, int* total) {
    const int tid = threadIdx.x, nt = blockDim.x;
    scratch[tid] = value;
    __syncthreads();
    for (int offset = 1; offset < nt; offset <<= 1) {
        const int add = tid >= offset ? scratch[tid - offset] : 0;
        __syncthreads();
        scratch[tid] += add;
        __syncthreads();
    }
    const int inclusive = scratch[tid];
    *total = scratch[nt - 1];
    __syncthreads();
    return inclusive - value;
}

extern "C" __global__ void uf_decode(
    const int m, const int n,
    const int* __restrict__ row_ptr, const int* __restrict__ row_col,   // checks -> bits
    const int* __restrict__ col_ptr, const int* __restrict__ col_row,   // bits -> checks
    const int* __restrict__ perm,           // the bits by increasing weight (ties by index)
    const unsigned char* __restrict__ syndromes, const int num_shots,
    int* __restrict__ workspace,            // per block: see the layout below
    const long long ws_ints,                // ints of workspace per block
    const int* __restrict__ obs_ptr, const int* __restrict__ obs_col, const int num_obs,
    unsigned char* __restrict__ predicted,  // shots x num_obs
    unsigned char* __restrict__ status,     // 1: every cluster valid, 0: a cluster could not grow any more
    int* __restrict__ next_shot)
{
    __shared__ int shot_s, list_n, pivot_s, progress_s, scan_total;
    __shared__ int scratch[1024];
    const int tid = threadIdx.x, nt = blockDim.x;
    int* ws = workspace + (size_t)blockIdx.x * (size_t)ws_ints;
    int* check_owner = ws;                  // m: cluster (detection event) of a check, -1 none
    int* parent = check_owner + m;          // m: disjoint-set forest of clusters
    int* flags = parent + m;                // m: bit 0 active, bit 1 valid, of a cluster root
    int* size = flags + m;                  // m: bits per cluster
    int* list = size + m;                   // m: invalid clusters in processing order
    int* rows = list + m;                   // m: the solved cluster's checks
    int* rowidx = rows + m;                 // m: row order during elimination
    int* pivcol = rowidx + m;               // m: pivot column of each eliminated row
    int* bit_owner = pivcol + m;            // n
    int* cols = bit_owner + n;              // n: the solved cluster's bits in weight order
    int* colpos = cols + n;                 // n: column of a bit in the cluster matrix
    unsigned char* cand = (unsigned char*)(colpos + n);       // n
    unsigned char* dec = cand + n;                            // n
    unsigned int* mat = (unsigned int*)(colpos + n + (2 * n + 3) / 4);

    while (true) {
        if (tid == 0) shot_s = atomicAdd(next_shot, 1);
        __syncthreads();
        const int s = shot_s;
        __syncthreads();
        if (s >= num_shots) return;
        const unsigned char* syn = syndromes + (size_t)s * m;

        for (int i = tid; i < m; i += nt) {
            check_owner[i] = syn[i] ? i : -1;
            parent[i] = i;
            flags[i] = syn[i] ? 1 : 0;
        }
        for (int j = tid; j < n; j += nt) { bit_owner[j] = -1; dec[j] = 0; }
        __syncthreads();

        unsigned char ok = 1;
        while (true) {
            // Invalid active clusters, smallest first (ties by detection event)
            for (int i = tid; i < m; i += nt) size[i] = 0;
            __syncthreads();
            for (int j = tid; j < n; j += nt)
                if (bit_owner[j] >= 0) atomicAdd(&size[find_root(parent, bit_owner[j])], 1);
            __syncthreads();
            if (tid == 0) {
                int k = 0;
                for (int i = 0; i < m; i++) {
                    if ((flags[i] & 3) != 1) continue;   // active and not valid
                    int pos = k++;
                    while (pos > 0 && (size[list[pos - 1]] > size[i])) { list[pos] = list[pos - 1]; pos--; }
                    list[pos] = i;
                }
                list_n = k;
                progress_s = 0;
            }
            __syncthreads();
            const int k_list = list_n;
            if (k_list == 0) break;

            for (int q = 0; q < k_list; q++) {
                const int c = list[q];
                if (!(flags[c] & 1)) continue;   // merged into another cluster this round
                // Grow: every bit next to one of the cluster's checks. ldpc lists a bit once per such check and
                // adds the first bits_per_step (= n) entries of that list by weight, so a large cluster of a
                // high-degree code adds only its lightest candidates: count each bit's entries
                int* entries = cols;   // n ints, free until the solve below
                for (int j = tid; j < n; j += nt) entries[j] = 0;
                __syncthreads();
                int listed = 0;
                for (int i = tid; i < m; i += nt) {
                    const int o = check_owner[i];
                    if (o < 0 || find_root(parent, o) != c) continue;
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        const int j = row_col[e];
                        const int b = bit_owner[j];
                        if (b < 0 || find_root(parent, b) != c) { atomicAdd(&entries[j], 1); listed++; }
                    }
                }
                int total_listed;
                block_scan(listed, scratch, &total_listed);
                if (total_listed > n) {
                    // keep a bit if its first entry comes before the n-th in weight order
                    int before = 0;
                    for (int base = 0; base < n; base += nt) {
                        const int r = base + tid;
                        const int j = r < n ? perm[r] : 0;
                        const int count = r < n ? entries[j] : 0;
                        int total;
                        const int off = block_scan(count, scratch, &total);
                        if (r < n) cand[j] = count > 0 && before + off < n;
                        before += total;
                    }
                } else {
                    for (int j = tid; j < n; j += nt) cand[j] = entries[j] > 0;
                }
                __syncthreads();
                int grew = 0;
                for (int j = tid; j < n; j += nt) {
                    if (!cand[j]) continue;
                    grew = 1;
                    const int b = bit_owner[j];
                    if (b >= 0) {
                        const int rb = find_root(parent, b);
                        if (rb != c) { parent[rb] = c; flags[rb] &= ~1; }
                    }
                    bit_owner[j] = c;
                    for (int e = col_ptr[j]; e < col_ptr[j + 1]; e++) {
                        const int i = col_row[e];
                        const int o = check_owner[i];
                        if (o < 0) {
                            check_owner[i] = c;
                        } else {
                            const int ro = find_root(parent, o);
                            if (ro != c) { parent[ro] = c; flags[ro] &= ~1; }
                        }
                    }
                }
                if (__syncthreads_or(grew) && tid == 0) progress_s = 1;

                // Solve: rows = the cluster's checks (any order), columns = its bits by weight
                int nr = 0;
                {
                    for (int base = 0; base < m; base += nt) {
                        const int i = base + tid;
                        int mine = 0;
                        if (i < m) { const int o = check_owner[i]; mine = o >= 0 && find_root(parent, o) == c; }
                        int total;
                        const int off = block_scan(mine, scratch, &total);
                        if (mine) rows[nr + off] = i;
                        nr += total;
                    }
                }
                int nc = 0;
                for (int base = 0; base < n; base += nt) {
                    const int r = base + tid;
                    int j = -1, mine = 0;
                    if (r < n) { j = perm[r]; const int b = bit_owner[j]; mine = b >= 0 && find_root(parent, b) == c; }
                    int total;
                    const int off = block_scan(mine, scratch, &total);
                    if (mine) { cols[nc + off] = j; colpos[j] = nc + off; }
                    nc += total;
                }
                __syncthreads();
                const int W = (nc + 1 + 31) / 32;   // the last column holds the syndrome
                for (int r = tid; r < nr; r += nt) {
                    unsigned int* row = mat + (size_t)r * W;
                    for (int w = 0; w < W; w++) row[w] = 0;
                    const int i = rows[r];
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        const int j = row_col[e];
                        const int b = bit_owner[j];
                        if (b >= 0 && find_root(parent, b) == c) {
                            const int p = colpos[j];
                            row[p >> 5] |= 1u << (p & 31);
                        }
                    }
                    if (syn[i]) row[nc >> 5] |= 1u << (nc & 31);
                    rowidx[r] = r;
                }
                __syncthreads();
                int rank = 0;
                for (int col = 0; col < nc && rank < nr; col++) {
                    const int w = col >> 5;
                    const unsigned int bit = 1u << (col & 31);
                    if (tid == 0) pivot_s = nr;
                    __syncthreads();
                    for (int r = rank + tid; r < nr; r += nt)
                        if (mat[(size_t)rowidx[r] * W + w] & bit) atomicMin(&pivot_s, r);
                    __syncthreads();
                    const int p = pivot_s;
                    if (p == nr) continue;   // the column depends on earlier ones
                    if (tid == 0) {
                        const int t = rowidx[p]; rowidx[p] = rowidx[rank]; rowidx[rank] = t;
                        pivcol[rank] = col;
                    }
                    __syncthreads();
                    const unsigned int* prow = mat + (size_t)rowidx[rank] * W;
                    for (int r = tid; r < nr; r += nt) {
                        if (r == rank) continue;
                        unsigned int* row = mat + (size_t)rowidx[r] * W;
                        if (row[w] & bit)
                            for (int x = w; x < W; x++) row[x] ^= prow[x];
                    }
                    rank++;
                    __syncthreads();
                }
                int bad = 0;
                for (int r = rank + tid; r < nr; r += nt)
                    if (mat[(size_t)rowidx[r] * W + (nc >> 5)] & (1u << (nc & 31))) bad = 1;
                const int invalid = __syncthreads_or(bad);
                for (int k = tid; k < nc; k += nt) dec[cols[k]] = 0;
                __syncthreads();
                for (int r = tid; r < rank; r += nt)
                    if (mat[(size_t)rowidx[r] * W + (nc >> 5)] & (1u << (nc & 31))) dec[cols[pivcol[r]]] = 1;
                if (tid == 0) flags[c] = invalid ? 1 : 3;
                __syncthreads();
            }
            if (!progress_s) { ok = 0; break; }   // no invalid cluster can grow
            __syncthreads();
        }
        for (int o = tid; o < num_obs; o += nt) {
            unsigned char bit = 0;
            for (int k = obs_ptr[o]; k < obs_ptr[o + 1]; k++) bit ^= dec[obs_col[k]];
            predicted[(size_t)s * num_obs + o] = bit;
        }
        if (tid == 0) status[s] = ok;
        __syncthreads();
    }
}
