// Relay-BP on the GPU, one shot per thread block: a port of relay-bp 0.2.1's RelayDecoder<f64> with its
// MinSumBPDecoder (crates/relay_bp/src/bp/relay.rs and min_sum.rs, Apache-2.0, IBM), keeping its arithmetic in
// the same order (FMA contraction off), so that with the same memory strengths the corrections are relay-bp's.
//
// Per shot: memory strengths gamma0 on every variable, check-to-variable messages 0, variable-to-check messages
// and posteriors the priors' log-likelihood ratios; pre_iter iterations of memory min-sum BP. Then up to num_sets
// legs: new memory strengths, check-to-variable messages 0, variable-to-check messages the priors, posteriors kept
// from the previous leg ("relayed"), up to set_max_iter iterations. The converged leg with the lowest weight
// (sum of the priors' log-likelihood ratios of its errors) is the answer; stop after stop_nconv legs converged;
// if none converges, the first leg's decoding.
//
// Memory strengths of the legs: an explicit table (row (leg mod rows), relay-bp's explicit_gammas), or uniform in
// [low, high) from a counter-based generator per (seed, shot, leg, variable). relay-bp draws them from one random
// stream that runs through the shots in order, which a parallel decoder cannot follow; the draws differ, their
// distribution is the same.
//
// Messages live on the check matrix's edges in CSR order (rows ascending, columns ascending within a row, as
// relay-bp's variable_to_check); col_edge lists each column's edges in ascending row order (its check_to_variable).

#define DBL_MAX_ 1.7976931348623157e308

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

extern "C" __global__ void relay_decode(
    const int m, const int n,
    const int* __restrict__ row_ptr, const int* __restrict__ row_col,
    const int* __restrict__ col_ptr, const int* __restrict__ col_edge,
    const double* __restrict__ lpr,            // n: log prior ratios (DBL_MAX for priors of 0), as relay-bp's
    const double* __restrict__ quality_lpr,    // n: the same where finite, 0 elsewhere (decoding quality)
    const unsigned char* __restrict__ syndromes, const int num_shots,
    const double gamma0, const int pre_iter, const int num_sets, const int set_max_iter, const int stop_nconv,
    const double* __restrict__ gamma_table, const int table_rows,   // explicit memory strengths, or rows 0
    const double gamma_low, const double gamma_high,
    const unsigned long long seed, const long long shot_offset,
    double* __restrict__ workspace, const int use_shared,
    unsigned char* __restrict__ best_out,      // num_shots x n: the chosen decoding
    unsigned char* __restrict__ converged, int* __restrict__ legs_run,
    int* __restrict__ next_shot)
{
    extern __shared__ double shared[];
    __shared__ int shot_s;
    __shared__ double quality_s;
    const int E = row_ptr[m];
    const size_t per_block = (size_t)2 * E + 2 * n + (2 * n + 7) / 8;
    double* base = use_shared ? shared : workspace + (size_t)blockIdx.x * per_block;
    double* v2c = base;
    double* c2v = base + E;
    double* posterior = base + 2 * E;
    double* gamma = base + 2 * E + n;
    unsigned char* dec = (unsigned char*)(base + 2 * E + 2 * n);
    const int tid = threadIdx.x, nt = blockDim.x;

    while (true) {
        if (tid == 0) shot_s = atomicAdd(next_shot, 1);
        __syncthreads();
        const int s = shot_s;
        __syncthreads();
        if (s >= num_shots) return;
        const unsigned char* syn = syndromes + (size_t)s * m;
        unsigned char* best = best_out + (size_t)s * n;

        // initialize_decoder: memory strengths gamma0, check messages 0, variable messages and posteriors = priors
        for (int j = tid; j < n; j += nt) {
            gamma[j] = gamma0;
            posterior[j] = lpr[j];
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) v2c[col_edge[k]] = lpr[j];
        }
        for (int e = tid; e < E; e += nt) c2v[e] = 0.0;
        __syncthreads();

        double best_pm = DBL_MAX_;
        int num_conv = 0, legs = 0;
        for (int leg = 0; leg <= num_sets; leg++) {
            if (leg > 0) {
                // init_next_set: new memory strengths; check messages 0, variable messages = priors
                for (int j = tid; j < n; j += nt) {
                    double g;
                    if (table_rows > 0) {
                        g = gamma_table[(size_t)(leg % table_rows) * n + j];
                    } else {
                        const unsigned long long key = splitmix64(seed ^ splitmix64(
                            (unsigned long long)(shot_offset + s) * 0x100000001B3ULL + (unsigned long long)leg));
                        const unsigned long long r = splitmix64(key + (unsigned long long)j);
                        const double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
                        g = gamma_low + (gamma_high - gamma_low) * u;
                    }
                    gamma[j] = g;
                    for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) v2c[col_edge[k]] = lpr[j];
                }
                for (int e = tid; e < E; e += nt) c2v[e] = 0.0;
                __syncthreads();
            }
            const int max_iter = leg == 0 ? pre_iter : set_max_iter;
            int success = 0;
            for (int it = 0; it < max_iter; it++) {
                // compute_check_to_variable (alpha 1)
                for (int i = tid; i < m; i += nt) {
                    bool accumulated_sign = syn[i] == 1;
                    int min_ind = 0;
                    double min_message = DBL_MAX_, second_min_message = DBL_MAX_;
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        const double v = v2c[e];
                        accumulated_sign ^= (bool)signbit(v);
                        const double a = fabs(v);
                        if (a <= min_message) {
                            second_min_message = min_message;
                            min_message = a;
                            min_ind = row_col[e];
                        } else if (a <= second_min_message) {
                            second_min_message = a;
                        }
                    }
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
                        const bool sign = accumulated_sign ^ (bool)signbit(v2c[e]);
                        const double mn = row_col[e] != min_ind ? min_message : second_min_message;
                        double msg = 1.0 * mn;
                        if (sign) msg = -msg;
                        c2v[e] = msg;
                    }
                }
                __syncthreads();
                // compute_variable_to_check with the memory prior, then hard decisions
                for (int j = tid; j < n; j += nt) {
                    double sum;
                    if (lpr[j] == DBL_MAX_) {
                        sum = lpr[j];
                    } else {
                        const double prior_component = (lpr[j] / 1.0) * (1.0 - gamma[j]);
                        const double posterior_component = (posterior[j] / 1.0) * gamma[j];
                        sum = prior_component + posterior_component;
                    }
                    for (int k = col_ptr[j]; k < col_ptr[j + 1]; k++) {
                        const int e = col_edge[k];
                        v2c[e] = sum;
                        sum += c2v[e];
                    }
                    posterior[j] = sum;
                    sum = 0.0;
                    for (int k = col_ptr[j + 1] - 1; k >= col_ptr[j]; k--) {
                        const int e = col_edge[k];
                        v2c[e] += sum;
                        sum += c2v[e];
                    }
                    dec[j] = posterior[j] <= 0.0 ? 1 : 0;
                }
                __syncthreads();
                int bad = 0;
                for (int i = tid; i < m && !bad; i += nt) {
                    unsigned char parity = 0;
                    for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) parity ^= dec[row_col[e]];
                    bad = parity != syn[i];
                }
                if (!__syncthreads_or(bad)) { success = 1; break; }
            }
            legs = leg + 1;
            if (success) {
                // decoding quality: the priors' log-likelihood ratios of the errors, summed in index order
                if (tid == 0) {
                    double q = 0.0;
                    for (int j = 0; j < n; j++) if (dec[j]) q += quality_lpr[j];
                    quality_s = q;
                }
                __syncthreads();
                const double pm = quality_s;
                num_conv++;
                if (leg == 0 || pm < best_pm) {
                    for (int j = tid; j < n; j += nt) best[j] = dec[j];
                    best_pm = pm;
                }
                __syncthreads();
                if (num_conv >= stop_nconv) break;
            } else if (leg == 0) {
                for (int j = tid; j < n; j += nt) best[j] = dec[j];
                __syncthreads();
            }
        }
        if (tid == 0) { converged[s] = num_conv > 0; legs_run[s] = legs; }
        __syncthreads();
    }
}
