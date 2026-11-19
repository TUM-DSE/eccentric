// Belief propagation on the GPU, one shot per thread block: a port of the parallel (flooding) schedule of
// ldpc 2.4.1 (src_cpp/bp.hpp, BpDecoder::bp_decode_parallel, MIT licence, Joschka Roffe) that keeps its
// arithmetic in the same order, so that with FMA contraction off (--fmad=false) every message, log-likelihood
// ratio and hard decision is the same double as on the CPU (minimum-sum; product-sum goes through CUDA's
// tanh/log, which can differ from glibc's in the last bit).
//
// Messages live on the edges of the check matrix, numbered in CSR order (row by row, columns ascending, as
// ldpc's row iteration); col_edge lists each column's edges in ascending row order (ldpc's column iteration).
// Blocks take shots from a shared counter, so a shot that runs to the iteration limit holds one block only.

#define DBL_MAX_ 1.7976931348623157e308

extern "C" __global__ void bp_parallel(
    const int m, const int n,
    const int* __restrict__ row_ptr, const int* __restrict__ row_col,
    const int* __restrict__ col_ptr, const int* __restrict__ col_edge,
    const double* __restrict__ llr0,          // n channel LLRs shared by all shots, or nullptr
    const double* __restrict__ probs,         // num_shots x n channel probabilities, or nullptr
    const unsigned char* __restrict__ syndromes,  // num_shots x m
    const int num_shots, const int max_iter, const int method, const double ms_scaling,
    double* __restrict__ workspace,           // gridDim.x x 3 x num_edges (+ n) doubles when not in shared memory
    const int edges, const int use_shared, const int zero_shortcut,
    unsigned char* __restrict__ decoding,     // num_shots x n
    double* __restrict__ llr_out,             // num_shots x n, or nullptr
    unsigned char* __restrict__ converged, int* __restrict__ iterations,
    int* __restrict__ next_shot)
{
    extern __shared__ double shared[];
    __shared__ int shot_s;
    double* base = use_shared ? shared : workspace + (size_t)blockIdx.x * (size_t)(2 * edges + n);
    double* b2c = base;
    double* c2b = base + edges;
    double* llr_init = base + 2 * edges;   // this shot's channel LLRs
    const int tid = threadIdx.x, nt = blockDim.x;

    while (true) {
        if (tid == 0) shot_s = atomicAdd(next_shot, 1);
        __syncthreads();
        const int s = shot_s;
        __syncthreads();
        if (s >= num_shots) return;
        const unsigned char* syn = syndromes + (size_t)s * m;
        unsigned char* dec = decoding + (size_t)s * n;

        // ldpc's decoders return zeros for a zero syndrome without running BP (zero_shortcut; the LLRs are then
        // set to 0, where ldpc keeps those of its previous decode)
        int nonzero = 0;
        for (int i = tid; i < m; i += nt) nonzero |= syn[i];
        if (!__syncthreads_or(nonzero) && zero_shortcut) {
            for (int j = tid; j < n; j += nt) {
                dec[j] = 0;
                if (llr_out) llr_out[(size_t)s * n + j] = 0.0;
            }
            if (tid == 0) { converged[s] = 1; iterations[s] = 0; }
            continue;
        }

        // initialise_log_domain_bp
        for (int j = tid; j < n; j += nt) {
            double l;
            if (probs) {
                const double p = probs[(size_t)s * n + j];
                l = log((1 - p) / p);
            } else {
                l = llr0[j];
            }
            llr_init[j] = l;
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) b2c[col_edge[k]] = l;
        }
        __syncthreads();

        int it_done = max_iter, conv = 0;
        for (int it = 1; it <= max_iter; it++) {
            // check-to-bit messages
            if (method == 0) {
                for (int i = tid; i < m; i += nt) {
                    double temp = 1.0;
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        c2b[e] = temp;
                        temp *= tanh(b2c[e] / 2);
                    }
                    temp = 1;
                    for (int e = row_ptr[i + 1] - 1; e >= row_ptr[i]; e--) {
                        c2b[e] *= temp;
                        const int message_sign = syn[i] != 0u ? -1.0 : 1.0;
                        c2b[e] = message_sign * log((1 + c2b[e]) / (1 - c2b[e]));
                        temp *= tanh(b2c[e] / 2);
                    }
                }
            } else {
                double alpha = ms_scaling == 0.0 ? 1.0 - pow(2.0, -1.0 * it) : ms_scaling;
                for (int i = tid; i < m; i += nt) {
                    int total_sgn = syn[i];
                    double temp = DBL_MAX_;
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        if (b2c[e] <= 0) total_sgn += 1;
                        c2b[e] = temp;
                        const double a = fabs(b2c[e]);
                        if (a < temp) temp = a;
                    }
                    temp = DBL_MAX_;
                    for (int e = row_ptr[i + 1] - 1; e >= row_ptr[i]; e--) {
                        int sgn = total_sgn;
                        if (b2c[e] <= 0) sgn += 1;
                        if (temp < c2b[e]) c2b[e] = temp;
                        const int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;
                        c2b[e] *= message_sign * alpha;
                        const double a = fabs(b2c[e]);
                        if (a < temp) temp = a;
                    }
                }
            }
            __syncthreads();

            // log probability ratios and hard decisions
            for (int j = tid; j < n; j += nt) {
                double temp = llr_init[j];
                for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                    const int e = col_edge[k];
                    b2c[e] = temp;
                    temp += c2b[e];
                }
                if (llr_out) llr_out[(size_t)s * n + j] = temp;
                dec[j] = temp <= 0 ? 1 : 0;
            }
            __syncthreads();

            // candidate syndrome against the syndrome
            int bad = 0;
            for (int i = tid; i < m && !bad; i += nt) {
                unsigned char parity = 0;
                for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) parity ^= dec[row_col[e]];
                bad = parity != syn[i];
            }
            if (!__syncthreads_or(bad)) { conv = 1; it_done = it; break; }

            // bit-to-check messages
            for (int j = tid; j < n; j += nt) {
                double temp = 0;
                for (int k = col_ptr[j + 1] - 1; k >= col_ptr[j]; k--) {
                    const int e = col_edge[k];
                    b2c[e] += temp;
                    temp += c2b[e];
                }
            }
            __syncthreads();
        }
        if (tid == 0) { converged[s] = conv; iterations[s] = it_done; }
        __syncthreads();
    }
}
