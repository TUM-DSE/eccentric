// GARI (normalized min-sum with a serial schedule) on the GPU, one shot per thread block: a port of
// ldpc_dec_msaa_quantum_serial_big_matrix_c in ../gari/hbp_decoder_v2.c (gari-nms, Apache-2.0) as GARI calls it
// (one ensemble member, iteration-based selection, early stopping on the second block's checks).
//
// The serial schedule processes rows one after another, each reading and updating the posteriors (lambda) of
// its variables. Rows that share no variable commute, so the host splits each iteration's row order into levels
// (a row goes one level after the last earlier row it shares a variable with): the rows of a level run in
// parallel and every variable sees the same updates in the same order as on the CPU, with the same arithmetic
// (FMA contraction off), so the decoder's output is the CPU's bit for bit.
//
// The C code keeps two message arrays (etar_old, etar) and copies one into the other after each iteration; each
// row's messages are read and then written only by that row once per iteration, so one array does the same.

#define INF_ __longlong_as_double(0x7ff0000000000000LL)

extern "C" __global__ void gari_decode(
    const int R, const int N,
    const int* __restrict__ row_ptr, const int* __restrict__ row_var,
    const double* __restrict__ llr,                 // N
    const signed char* __restrict__ syndromes,      // shots x R, +1/-1 (1 beyond the detectors)
    const int num_shots, const int n_loop, const double alpha,
    const int* __restrict__ sched_rows,             // (n_loop + 1) x R: each iteration's rows, level by level
    const int* __restrict__ level_start,            // each level's rows: sched_rows[level_start .. level_end)
    const int* __restrict__ level_end,
    const int* __restrict__ iter_level,             // (n_loop + 2): each iteration's first level
    const int check_lo, const int check_hi,         // rows of the convergence check (dx rows .. hshape0)
    const int nx,
    const int* __restrict__ obs_ptr, const int* __restrict__ obs_col,  // observables over the dz block
    const int num_obs,
    double* __restrict__ workspace, const int use_shared,
    signed char* __restrict__ predicted,            // shots x num_obs
    int* __restrict__ iterations,
    int* __restrict__ next_shot)
{
    extern __shared__ double shared[];
    __shared__ int shot_s;
    const int E = row_ptr[R];
    double* base = use_shared ? shared : workspace + (size_t)blockIdx.x * (size_t)(E + N + (N + 7) / 8);
    double* lambda = base;
    double* etar = base + N;
    unsigned char* hdec = (unsigned char*)(base + N + E);
    const int tid = threadIdx.x, nt = blockDim.x;

    while (true) {
        if (tid == 0) shot_s = atomicAdd(next_shot, 1);
        __syncthreads();
        const int s = shot_s;
        __syncthreads();
        if (s >= num_shots) return;
        const signed char* syn = syndromes + (size_t)s * R;

        for (int j = tid; j < N; j += nt) {
            lambda[j] = fmin(fmax(llr[j], -10000.0), 10000.0);
            hdec[j] = 0;
        }
        for (int e = tid; e < E; e += nt) etar[e] = 0.0;
        __syncthreads();

        int loop;
        for (loop = 0; loop <= n_loop; ++loop) {
            for (int lv = iter_level[loop]; lv < iter_level[loop + 1]; lv++) {
                for (int k = level_start[lv] + tid; k < level_end[lv]; k += nt) {
                    const int r = sched_rows[k];
                    const int lo = row_ptr[r], hi = row_ptr[r + 1];
                    double min1 = INF_, min2 = INF_;
                    int posm = -1;
                    int sign_t = 1;
                    for (int e = lo; e < hi; e++) {
                        const int idx = row_var[e];
                        lambda[idx] -= etar[e];
                        const double abs_val = fabs(lambda[idx]);
                        if (abs_val < min1) {
                            min2 = min1;
                            min1 = abs_val;
                            posm = e - lo;
                        } else if (abs_val < min2) {
                            min2 = abs_val;
                        }
                        sign_t = lambda[idx] < 0 ? -sign_t : sign_t;
                    }
                    for (int e = lo; e < hi; e++) {
                        const int idx = row_var[e];
                        const double pr = (e - lo == posm ? min2 : min1) * alpha;
                        const int sign = lambda[idx] < 0 ? -sign_t : sign_t;
                        etar[e] = sign * pr * syn[r];
                    }
                    for (int e = lo; e < hi; e++) {
                        const int idx = row_var[e];
                        lambda[idx] += etar[e];
                        hdec[idx] = lambda[idx] < 0 ? 1 : 0;
                    }
                }
                __syncthreads();
            }
            // Early stopping: the parity of the checks check_lo .. check_hi
            int bad = 0;
            for (int r = check_lo + tid; r < check_hi && !bad; r += nt) {
                int sign_sum = 0;
                for (int e = row_ptr[r]; e < row_ptr[r + 1]; e++) sign_sum += hdec[row_var[e]];
                bad = sign_sum % 2 != ((syn[r] - 1) / -2);
            }
            if (!__syncthreads_or(bad)) break;
        }
        // GARI's selection with one member: a shot counts as decoded if it converged before the last iteration;
        // otherwise its correction is zero
        for (int o = tid; o < num_obs; o += nt) {
            signed char bit = 0;
            if (loop < n_loop)
                for (int k = obs_ptr[o]; k < obs_ptr[o + 1]; k++) bit ^= hdec[nx + obs_col[k]];
            predicted[(size_t)s * num_obs + o] = bit;
        }
        if (tid == 0) iterations[s] = loop;
        __syncthreads();
    }
}
