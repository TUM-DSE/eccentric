/* GARI normalized min-sum decoder, from https://github.com/astra-decoders/gari-nms (commit 6380d52e76d8c9cb0b4eedf3e8d2429b24cef78d),
 * my_decoders/hbp_decoder_v2.c, unchanged. Copyright the gari-nms authors (A. S. Maan, F. Garcia-Herrero,
 * A. Paler, V. Savin), Apache License 2.0 (see LICENSE in this directory). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Phase 3 NUMA Awareness: Add NUMA-related headers
#ifdef _GNU_SOURCE
#include <sched.h>
#include <numa.h>
#endif

// Memory safety constants to prevent segmentation faults
#define MAX_MEMORY_PER_THREAD_GB 2  // Limit to 4GB per thread
#define MAX_BATCH_SIZE 1000000      // Reasonable batch size limit
#define MAX_ENSEMBLE_SIZE 100       // Reasonable ensemble size limit

// Clip function similar to np.clip
static inline double clip(double x, double min, double max) {
    return fmin(fmax(x, min), max);
}

// Function to allocate contiguous memory for 2D arrays
static inline double* alloc_2d_contiguous(int rows, int cols) {
    return calloc(rows * cols, sizeof(double));
}

// Function to allocate memory for a 1D integer array
static inline int* alloc_1d_int(int size) {
    return malloc(size * sizeof(int));
}

// Simulated permutation function
//static inline void random_permutation(int* array, int n, int seed) {
//    srand(seed);  // Set random seed
//    for (int i = 0; i < n; i++) {
//        int j = rand() % (i + 1);
//        int temp = array[i];
//        array[i] = array[j];
//        array[j] = temp;
//    }
//}
static inline void random_permutation(int* array, int n, int seed) {
    if (n <= 1) return;

    // Fast xorshift PRNG - much faster than rand()
    uint32_t state = (uint32_t)seed;
    if (state == 0) state = 1;

    for (int i = n - 1; i > 0; i--) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        int j = state % (i + 1);
//        printf("Swapping index %d and index %d\n", i, j);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Phase 3 NUMA Awareness: Thread affinity and memory locality function
#ifdef _GNU_SOURCE
static inline void set_thread_affinity_numa() {
    #ifdef _OPENMP
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        // Set thread affinity to specific CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        
        // Set NUMA memory allocation preference to local node
        if (numa_available() >= 0) {
            int cpu_node = numa_node_of_cpu(thread_id);
            if (cpu_node >= 0) {
                numa_set_preferred(cpu_node);
            }
        }
    }
    #endif
}
#else
// Fallback for non-GNU systems
static inline void set_thread_affinity_numa() {
    // No-op on systems without NUMA support
}
#endif

// Main LDPC decoder function
__attribute__((visibility("default")))
int ldpc_dec_msaa_quantum_serial_big_matrix_c(
    double* llr, int Nloop,
    int* Hrows, int Hrows_r, int Hrows_c,
    int* Hcols, int Hcols_r, int Hcols_c,
    double alpha, int8_t* Syn_x,
    int random_order, int hshape0, int dxshape0,
    int custom_random_schedule_HBP,
    int early_stopping,
    int8_t* Hdec_out, double* lambda_out
) {
    // Initialize variables
    //int i, j, ;
    int loop;
    int N = Hcols_r;

    double* etar = alloc_2d_contiguous(Hrows_r, Hrows_c);
    double* etar_old = alloc_2d_contiguous(Hrows_r, Hrows_c);
    int8_t* Hdec = malloc(N * sizeof(int8_t));
    double* lambda = malloc(N * sizeof(double));

//    double alpha_itr = alpha;

//    int n_alpha = 4;
//    double* alpha_arr = malloc(n_alpha * sizeof(double));
//    alpha_arr[0] = 0.9375;
//    alpha_arr[1] = 0.96875;
//    alpha_arr[2] = 0.984375;
//    alpha_arr[3] = 0.9921875;


    // Add error checking for memory allocations
    if (!etar || !etar_old || !Hdec || !lambda) {
        // Clean up any successful allocations
        if (etar) free(etar);
        if (etar_old) free(etar_old);
        if (Hdec) free(Hdec);
        if (lambda) free(lambda);
        // Return error indication
        return Nloop;
    }
    
    memcpy(lambda, llr, N * sizeof(double));

    // Pre-compute clipped lambda values
//    #pragma GCC ivdep
    for (int i = 0; i < N; ++i) {
        lambda[i] = clip(lambda[i], -10000.0, 10000.0);
    }

    int* row_indices = malloc(Hrows_r * sizeof(int));
    int* row_indices_bottom = malloc((Hrows_r - hshape0) * sizeof(int));
    int* row_indices_a = malloc(dxshape0 * sizeof(int));
    int* row_indices_b = malloc((hshape0 - dxshape0) * sizeof(int));
    int* row_indices_ab = malloc(hshape0 * sizeof(int));
    
    // Add error checking for row_indices allocations
    if (!row_indices || !row_indices_bottom || !row_indices_a || !row_indices_b || !row_indices_ab) {
        // Clean up all allocations
        if (row_indices) free(row_indices);
        if (row_indices_bottom) free(row_indices_bottom);
        if (row_indices_a) free(row_indices_a);
        if (row_indices_b) free(row_indices_b);
        if (row_indices_ab) free(row_indices_ab);
        free(etar);
        free(etar_old);
        free(Hdec);
        free(lambda);
        return Nloop;
    }

//    #pragma GCC ivdep
    for (int i = 0; i < Hrows_r; ++i){
            row_indices[i] = i;
            if (i < Hrows_r - hshape0) {row_indices_bottom[i] = i + hshape0;}
            if (i < dxshape0) {row_indices_a[i] = i;}
            if (i < hshape0 - dxshape0){row_indices_b[i] = i + dxshape0;}
            if (i < hshape0) {row_indices_ab[i] = i;}
    }

    for (loop = 0; loop <= Nloop; ++loop)
    {
        switch (custom_random_schedule_HBP) {
            case 1: { // bottom-AB
                    if (random_order) {
//                    random_permutation(row_indices_bottom, Hrows_r - hshape0, random_order+loop);
//                    for(int k = 0; k < Hrows_r - hshape0; k++) printf("%d ", row_indices_bottom[k]);
                    random_permutation(row_indices_ab, hshape0, random_order+loop);
                    memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0), row_indices_ab, hshape0 * sizeof(int));
                    }
                    if (loop ==0){
                        memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                        memcpy(row_indices + (Hrows_r - hshape0), row_indices_ab, hshape0 * sizeof(int));
                    }

                    break;
                    }
            case 2: { // bottom-A-B
                    if (random_order){
//                    random_permutation(row_indices_bottom, Hrows_r - hshape0, random_order+loop);
                    random_permutation(row_indices_a, dxshape0, random_order+loop);
                    random_permutation(row_indices_b, hshape0 - dxshape0, random_order+loop);
                    memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0), row_indices_a, dxshape0 * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0) + dxshape0, row_indices_b, (hshape0 - dxshape0) * sizeof(int));
                    }
                    if (loop==0){
                    memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0), row_indices_a, dxshape0 * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0) + dxshape0, row_indices_b, (hshape0 - dxshape0) * sizeof(int));
                    }

                    break;
                    }
            case 3: { // bottom-B-A
                    if (random_order){
//                    random_permutation(row_indices_bottom, Hrows_r - hshape0, random_order+loop);
                    random_permutation(row_indices_b, hshape0 - dxshape0, random_order+loop);
                    random_permutation(row_indices_a, dxshape0, random_order+loop);

                    memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0), row_indices_b, (hshape0 - dxshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - dxshape0), row_indices_a, dxshape0 * sizeof(int));
                    }

                    if (loop==0){
                    memcpy(row_indices, row_indices_bottom, (Hrows_r - hshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - hshape0), row_indices_b, (hshape0 - dxshape0) * sizeof(int));
                    memcpy(row_indices + (Hrows_r - dxshape0), row_indices_a, dxshape0 * sizeof(int));
                    }

                    break;
                    }
            default: { // default schedule
                    if (random_order) random_permutation(row_indices, Hrows_r, random_order+loop);
                    }
        }
//        for(int k = 0; k < Hrows_r; k++) printf("%d ", row_indices[k]);

        // Process each row in row_indices
////        #pragma GCC ivdep
        for (int irHR_idx = 0; irHR_idx < Hrows_r; ++irHR_idx) {
            int irHR = row_indices[irHR_idx];
            double min1 = INFINITY, min2 = INFINITY;
            int posm = -1;
            int sign_t = 1;

            // Check node update using SIMD directives
////            #pragma GCC ivdep
            for (int icHR = 0; icHR < Hrows_c; ++icHR) {
                int idx = Hrows[irHR * Hrows_c + icHR];
                if (idx == 0) break;
                idx--;

                lambda[idx] -= etar_old[irHR * Hrows_c + icHR];
                double abs_val = fabs(lambda[idx]);

                if (abs_val < min1) {
                    min2 = min1;
                    min1 = abs_val;
                    posm = icHR;
                } else if (abs_val < min2) {
                    min2 = abs_val;
                }

                sign_t = lambda[idx] < 0 ? -sign_t : sign_t;
            }

////            #pragma GCC ivdep
            for (int icHR = 0; icHR < Hrows_c; ++icHR) {
                int idx = Hrows[irHR * Hrows_c + icHR];
                if (idx == 0) break;
                idx--;

//                if (alpha>1.0){
//                    // Use fast xorshift PRNG instead of slow rand()
//                    static uint32_t alpha_state = 0;
//                    if (alpha_state == 0) alpha_state = (uint32_t)(loop + random_order + 1);
//                    alpha_state ^= alpha_state << 13;
//                    alpha_state ^= alpha_state >> 17;
//                    alpha_state ^= alpha_state << 5;
//                    int alpha_i = alpha_state % n_alpha;
//                    alpha_itr = alpha_arr[alpha_i];
//                }
//                  if (irHR < hshape0){
//                    alpha_itr = 1.0;
//                  }
//                printf("%f ", alpha_itr);
//                double pr = (icHR == posm ? min2 : min1) * alpha_itr;
                double pr = (icHR == posm ? min2 : min1) * alpha;
                int sign = lambda[idx] < 0 ? -sign_t : sign_t;

                etar[irHR * Hrows_c + icHR] = sign * pr * Syn_x[irHR];
            }

            // Message passing update
////            #pragma GCC ivdep
            for (int icHR = 0; icHR < Hrows_c; ++icHR) {
                int idx = Hrows[irHR * Hrows_c + icHR];
                if (idx == 0) break;
                idx--;

                lambda[idx] += etar[irHR * Hrows_c + icHR];
                Hdec[idx] = lambda[idx] < 0 ? 1 : 0;
            }
        }

        // Copy updated etar values
        memcpy(etar_old, etar, Hrows_r * Hrows_c * sizeof(double));

        // Check parity
        int parity_t = 0;
        // Vectorized early stopping check

        int mylen = Hrows_r;
        int mylen_is = 0;
        if(early_stopping){
            mylen = hshape0;
//            mylen = dxshape0;
            mylen_is = dxshape0;
        }

        int *parity_t_vec = calloc(mylen, sizeof(int));  // For parity checking results
        int *sign_t_vec = calloc(Hrows_c, sizeof(int));   // For inner loop calculations
//            #pragma GCC ivdep
//        for (int irHR_idx = 0; irHR_idx < mylen; ++irHR_idx) {
        for (int irHR_idx=mylen_is; irHR_idx < mylen; ++irHR_idx) {
            memset(sign_t_vec, 0, Hrows_c * sizeof(int));

//                #pragma GCC ivdep
            for (int icHR = 0; icHR < Hrows_c; ++icHR) {
                int idx = Hrows[irHR_idx * Hrows_c + icHR];
                if (idx == 0) break;
                idx--;
                sign_t_vec[icHR] = Hdec[idx];
            }

            int sign_sum = 0;
            for (int icHR = 0; icHR < Hrows_c; ++icHR) {
                sign_sum += sign_t_vec[icHR];
            }

            if (sign_sum % 2 != ((Syn_x[irHR_idx] - 1) / -2)) {
                parity_t_vec[irHR_idx] = 1;
            }
        }

        // Sum the parity_t_vec to get total parity violations
//        for (int irHR_idx = 0; irHR_idx < mylen; ++irHR_idx) {
        for (int irHR_idx=mylen_is; irHR_idx < mylen; ++irHR_idx) {
            parity_t += parity_t_vec[irHR_idx];
        }
        free(sign_t_vec);
        free(parity_t_vec);

        // Terminate if parity check passes
        if (parity_t == 0) break;
    }

    // Transfer results out
    memcpy(Hdec_out, Hdec, N * sizeof(int8_t));
    memcpy(lambda_out, lambda, N * sizeof(double));

//    free(alpha_arr);
    free(row_indices);
    free(row_indices_bottom);
    free(row_indices_a);
    free(row_indices_b);
    free(row_indices_ab);
    free(lambda);
    free(Hdec);
    free(etar);
    free(etar_old);

    return loop;
}



// Batched Ensemble Decoding Function
// Combines batching with ensemble decoding, dividing threads between data parallelism and ensemble size
__attribute__((visibility("default")))
void ldpc_dec_msaa_quantum_serial_big_matrix_c_ensemble_batched_ler(
    double* llr_batch, double* llr_ab_batch, int batch_size, int N, int Nloop,
    int* Hrows, int Hrows_r, int Hrows_c,
    int* Hcols, int Hcols_r, int Hcols_c,
    double alpha, int8_t* Syn_x_batch,
    int random_order, int hshape0, int dxshape0,
    int custom_random_schedule_HBP,
    int early_stopping, int num_threads,
    int ensemble_size, char prob_or_itr,  // 'w' for weight-based, 'i' for iteration-based
    int nx, int nz, int mx, int m,
    int8_t* dz, int dz_rows, int dz_cols,
    int8_t* l_dz, int l_dz_rows, int l_dz_cols,
    int8_t* logical_errors_out, int* iterations_out
) {
    // Add comprehensive parameter validation to prevent segmentation fault
    if (batch_size <= 0 || batch_size > MAX_BATCH_SIZE) {
        printf("ERROR: Invalid batch_size=%d (max allowed: %d)\n", batch_size, MAX_BATCH_SIZE);
        return;
    }
    
    if (N <= 0 || N > 1000000) {  // Reasonable limit for N
        printf("ERROR: Invalid N=%d\n", N);
        return;
    }
    
    if (ensemble_size <= 0 || ensemble_size > MAX_ENSEMBLE_SIZE) {
        printf("ERROR: Invalid ensemble_size=%d (max allowed: %d)\n", ensemble_size, MAX_ENSEMBLE_SIZE);
        return;
    }
    
    if (!llr_batch || !llr_ab_batch || !Syn_x_batch || !logical_errors_out || !iterations_out) {
        printf("ERROR: Null pointer parameters detected\n");
        return;
    }
    
    // Validate matrix dimensions for consistency
    if (nx < 0 || nz < 0 || mx < 0 || m < 0) {
        printf("ERROR: Negative matrix dimensions\n");
        return;
    }
    
    if (nx + nz > N) {
        printf("ERROR: Matrix dimension inconsistency: nx+nz=%d > N=%d\n", nx + nz, N);
        return;
    }
    
//    printf("DEBUG: Function entry successful - batch_size=%d, N=%d, ensemble_size=%d\n",batch_size, N, ensemble_size);
    
    // Set number of OpenMP threads
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    #endif
    
//    printf("DEBUG: OpenMP threads set successfully\n");
    
    // Pre-validate memory requirements to prevent segmentation fault
    size_t ensemble_memory_per_thread = (size_t)ensemble_size * (size_t)N * sizeof(int8_t);
    
    // Check for integer overflow in memory calculation
    if (ensemble_size > 0 && N > 0 && ensemble_memory_per_thread / (size_t)ensemble_size != (size_t)N * sizeof(int8_t)) {
        printf("ERROR: Integer overflow in memory calculation\n");
        return;
    }
    
    size_t total_estimated_memory = ensemble_memory_per_thread * (size_t)batch_size;
    size_t max_allowed_memory = (size_t)MAX_MEMORY_PER_THREAD_GB * 1024ULL * 1024ULL * 1024ULL;
    
    printf("DEBUG: Memory validation - per_thread=%zu MB, total=%zu MB\n",
           ensemble_memory_per_thread / (1024 * 1024), total_estimated_memory / (1024 * 1024));
    
    // Check for potential memory overflow
    if (ensemble_memory_per_thread > max_allowed_memory) {
        printf("ERROR: Memory requirement too large per thread: %zu MB (max: %zu MB)\n",
               ensemble_memory_per_thread / (1024 * 1024), max_allowed_memory / (1024 * 1024));
        printf("Consider reducing ensemble_size (%d) or N (%d)\n", ensemble_size, N);
        return;
    }
    
    // If memory usage is high, reduce thread count for safety
    if (ensemble_memory_per_thread > max_allowed_memory / 4) {
        printf("WARNING: High memory usage detected, reducing thread count for safety\n");
        #ifdef _OPENMP
        if (num_threads > 1) {
            omp_set_num_threads((num_threads + 1) / 2);  // Use integer division instead of ceil()
        }
        #endif
    }
    
//    printf("DEBUG: About to start parallel loop\n");
    
    // Phase 1 Optimization: Pre-allocate thread-local storage for complex batched ensemble processing
    int max_threads = num_threads > 0 ? num_threads : omp_get_max_threads();
    int8_t*** pred_err_sr_threads = malloc(max_threads * sizeof(int8_t**));
    int** iter_cov_threads = malloc(max_threads * sizeof(int*));
    double** lambda_out_threads = malloc(max_threads * sizeof(double*));
    
    for (int t = 0; t < max_threads; t++) {
        pred_err_sr_threads[t] = malloc(sizeof(int8_t*));
        pred_err_sr_threads[t][0] = malloc(ensemble_size * N * sizeof(int8_t));
        iter_cov_threads[t] = malloc(ensemble_size * sizeof(int));
        lambda_out_threads[t] = malloc(N * sizeof(double));
    }
    
    // Phase 1 Optimization: Use static scheduling with optimal chunk size
    int chunk_size = (batch_size + max_threads - 1) / max_threads;
    
    // Phase 3 NUMA Awareness: Set thread affinity before parallel processing
    set_thread_affinity_numa();
    
    // Simplified thread management - use single level parallelism
    // Parallel processing of batch
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static, chunk_size)
    #endif
    for (int shot = 0; shot < batch_size; shot++) {
        // Add bounds check for shot index
        if (shot >= batch_size || shot < 0) {
            printf("ERROR: Invalid shot index %d\n", shot);
            continue;
        }
        
        // Use same 1D LLR arrays for all shots
        double* llr_shot = llr_batch;  // Same LLR values for each shot
        double* llr_ab_shot = llr_ab_batch;  // Same LLR values for each shot
        
        // Bounds check for syndrome array access
        if ((size_t)shot * (size_t)Hrows_r >= (size_t)batch_size * (size_t)Hrows_r) {
            printf("ERROR: Syndrome array bounds exceeded for shot %d\n", shot);
            continue;
        }
        int8_t* Syn_x_shot = Syn_x_batch + shot * Hrows_r;

        // Use pre-allocated thread-local arrays (eliminates malloc contention)
        int thread_id = omp_get_thread_num();
        int8_t* pred_err_sr_ens = pred_err_sr_threads[thread_id][0];
        int* iter_cov_ens = iter_cov_threads[thread_id];
        
        // Check for allocation errors from pre-allocation phase
        if (!pred_err_sr_ens || !iter_cov_ens) {
            // Set safe error indicators for this shot and continue
            if (shot * l_dz_rows + l_dz_rows <= batch_size * l_dz_rows) {
                for (int i = 0; i < l_dz_rows; i++) {
                    logical_errors_out[shot * l_dz_rows + i] = 0;
                }
            }
            if (shot < batch_size) {
                iterations_out[shot] = Nloop;
            }
            continue;
        }
////        printf("DEBUG: parallel loop 2\n");
        // Run ensemble decodings sequentially within each thread
        for (int ens = 0; ens < ensemble_size; ens++) {
            int8_t* pred_err_sr_shot_ens = pred_err_sr_ens + ens * N;
            
            // Use pre-allocated thread-local array for lambda output
            double* lambda_out_thread = lambda_out_threads[thread_id];
            if (!lambda_out_thread) {
                // Set error indication for this ensemble and continue
                iter_cov_ens[ens] = Nloop;
                continue;
            }
////            printf("DEBUG: parallel loop en start\n");
            iter_cov_ens[ens] = ldpc_dec_msaa_quantum_serial_big_matrix_c(
                llr_shot, Nloop,
                Hrows, Hrows_r, Hrows_c,
                Hcols, Hcols_r, Hcols_c,
                alpha, Syn_x_shot,
                random_order * ensemble_size + Nloop * ens,  // Unique random seed per ensemble
                hshape0, dxshape0,
                custom_random_schedule_HBP,
                early_stopping,
                pred_err_sr_shot_ens, lambda_out_thread
            );
        }
////        printf("DEBUG: parallel loop ens end\n");
        // Find best ensemble for this shot
        int best_ensemble = -1;
        double best_score = INFINITY;
        int min_iter = Nloop;
        // Check if any ensemble converged
        int any_converged = 0;
        for (int ens = 0; ens < ensemble_size; ens++) {
            if (iter_cov_ens[ens] < Nloop) {
                any_converged = 1;
                min_iter = iter_cov_ens[ens];
                break;
            }
        }
        
        if (any_converged) {
            // Process converged ensembles
            for (int ens = 0; ens < ensemble_size; ens++) {
                if (iter_cov_ens[ens] < Nloop) {
                    int8_t* corr_dz = pred_err_sr_ens + ens * N + nx;
                    
                    if (prob_or_itr == 'w') {
                        // Weight-based selection: minimize (corr_dz @ llr_ab[nx:nx+nz])
                        double score = 0.0;
                        for (int j = 0; j < nz; j++) {
                            if (corr_dz[j]) {
                                score += llr_ab_shot[nx + j];
                            }
                        }
                        if (score < best_score) {
                            best_score = score;
                            best_ensemble = ens;
                        }
                    } else if (prob_or_itr == 'i') {
                        // Iteration-based selection: prefer minimum iterations
                        //Note: weight based selection if many decoders in ens converge on same iteration (which is min)
                        if (iter_cov_ens[ens] <= min_iter) {
                            min_iter = iter_cov_ens[ens];
                            double score = 0.0;
                            for (int j = 0; j < nz; j++) {
                                if (corr_dz[j]) {
                                    score += llr_ab_shot[nx + j];
                                }
                            }
                            if (score < best_score) {
                                best_score = score;
                                best_ensemble = ens;
                            }
                        }

                    }
                }
            }
        }

        
        // Calculate logical error from best ensemble
        if (best_ensemble >= 0) {
            int8_t* best_corr_dz = pred_err_sr_ens + best_ensemble * N + nx;
            
            // Calculate logical error: (l_dz @ corr_dz) % 2
            for (int i = 0; i < l_dz_rows; i++) {
                int8_t logical_bit = 0;
                for (int j = 0; j < l_dz_cols; j++) {
                    if (l_dz[i * l_dz_cols + j] != 0) {
                        logical_bit ^= (int8_t)best_corr_dz[j];
                    }
                }
                logical_errors_out[shot * l_dz_rows + i] = logical_bit;
            }
            iterations_out[shot] = iter_cov_ens[best_ensemble];
        } else {
            // No valid ensemble found, return zero correction
            for (int i = 0; i < l_dz_rows; i++) {
                logical_errors_out[shot * l_dz_rows + i] = 0;
            }
            iterations_out[shot] = Nloop;
        }
    }
    
    // Phase 1 Cleanup: Free pre-allocated thread-local storage for complex batched ensemble processing
    for (int t = 0; t < max_threads; t++) {
        free(pred_err_sr_threads[t][0]);
        free(pred_err_sr_threads[t]);
        free(iter_cov_threads[t]);
        free(lambda_out_threads[t]);
    }
    free(pred_err_sr_threads);
    free(iter_cov_threads);
    free(lambda_out_threads);
}