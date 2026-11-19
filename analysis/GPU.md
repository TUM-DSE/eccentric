# GPU decoders

ECCentric's decoders that parallelise well have GPU versions in `eccentric/decoders/gpu/`. Each is a decoder name
for `decode()` with the same arguments, the same settings as the CPU decoder of the same name without `_gpu`, and
the same return value (the logical error rate). Matching-type decoders (`mwpm`, `mwpm_corr`, `chromobius`,
`tesseract`, `mwpf`) stay on the CPU.

| GPU decoder | CPU decoder | on the GPU | on the CPU | agreement with the CPU decoder |
|---|---|---|---|---|
| `bplsd_gpu` | `bplsd` | BP (our kernel) | LSD for the shots BP does not converge on (ldpc's code) | identical on every shot |
| `belief_find_gpu` | `belief_find` | BP (our kernel) | union-find for the shots BP does not converge on (ldpc's code) | identical on every shot |
| `gari_gpu` | `gari` | GARI (our kernel) | — | identical on every shot |
| `hier_bposd_hard_gpu` | `hier_bposd_hard` | level stages and final BP (our kernel) | OSD for the shots the final BP does not converge on (ldpc) | identical on every shot tested |
| `hier_bposd_gpu` | `hier_bposd` | level stages and final BP (our kernel) | OSD as above | 99.0–99.4% of shots; LER within statistical error |
| `bposd_gpu`, `bposd_tuned_gpu` | `bposd`, `bposd_tuned` | BP and OSD-CS (NVIDIA nv-qldpc-decoder) | — | 98.1–100% of shots; LER within statistical error |
| `relay_bp_gpu` | `relay_bp` | Relay-BP (our kernel, port of relay-bp) | — | identical on every shot with the same memory strengths; with per-shot random ones (the default) 84–100% of shots, LER within statistical error |
| `union_find_gpu` | `union_find` | union-find with inversion (our kernel) | — | 54–99.7% of shots; LER differs by 0.1–2% (relative) on 5 of 7 circuits (see below) |

## How they work

All kernels decode one shot per thread block; blocks take shots from a shared counter, so a shot that runs to the
iteration limit occupies one block while the others move on. Kernels are compiled with NVRTC through CuPy on first
use, with fused multiply-adds off so that double arithmetic rounds as on the CPU.

- **BP** (`bp.cu`, `bp.py`): a port of ldpc 2.4.1's parallel-schedule BP (`src_cpp/bp.hpp`) that keeps its
  operations in the same order (prefix and suffix minima per check, the same summation order per variable). Min-sum
  BP gives the same log-likelihood ratios bit for bit, the same decisions and the same iteration counts as ldpc.
  Product-sum BP goes through CUDA's `tanh` and `log`, which differ from the C library's in the last bits
  (log-likelihood ratios differ by up to ~1e-11).
- **BP + CPU post-processing** (`ldpc_gpu.py`): BP runs on the GPU for a batch of shots; the shots it does not
  converge on go to ldpc's own LSD (`LsdDecoder`), union-find (`UnionFindDecoder`) or OSD (`BpOsdDecoder`) on the
  CPU, with the GPU's log-likelihood ratios. With identical ratios, these give the CPU decoders' corrections
  exactly. When BP fails on many shots, the CPU part dominates the time.
- **GARI** (`gari.cu`, `gari_gpu.py`): the C decoder's serial schedule updates one row at a time. Its row order
  depends only on the seed and the iteration, so the host computes it for every iteration (the same xorshift
  permutations) and splits it into levels of rows that share no variable; a level's rows run in parallel and each
  variable sees the same updates in the same order. Circuits tested need 12–25 levels per iteration.
- **Union-find** (`union_find.cu`): ldpc's `matrix_decode` (cluster growth by all candidate bits, merging, then
  Gaussian elimination over each cluster's bits ordered by weight).
- **Relay-BP** (`relay.cu`, `relay_gpu.py`): a port of relay-bp 0.2.1's `RelayDecoder<f64>` (memory min-sum BP
  and its legs) with its arithmetic in the same order, on relay-bp's own check matrices (including the biases of
  pruned errors) and with ECCentric's settings (relay-bp's sinter defaults: γ0 0.1, 60 pre-iterations, up to 60
  legs of 60 iterations, γ uniform in [−0.24, 0.66], stop after 5 converged legs, lowest-weight solution). Given
  the same memory strengths (relay-bp's `explicit_gammas`), it returns relay-bp's error vectors on every shot.
  relay-bp draws the legs' memory strengths from one random stream that runs through the shots in order, which a
  parallel decoder cannot follow; the kernel draws them per (seed, shot, leg, variable) from a counter-based
  generator instead: different draws, the same distribution.
- **nv-qldpc** (`nvqldpc.py`): BP-OSD through NVIDIA CUDA-Q QEC's `nv-qldpc-decoder` (closed source), with
  ECCentric's settings: min-sum BP with `bposd`'s or `bposd_tuned`'s iterations and scaling, OSD-CS of the same
  order. `relay_bp_gpu` first used nv-qldpc's Relay-BP too; it is kept as `nvqldpc.relay_bp_nvqldpc` for
  comparison (see below).

## Installing and running

    pip install --no-deps -r requirements-gpu.txt

`--no-deps` keeps `requirements.txt`'s numpy (cudaq-qec would install numpy 2; tested with numpy 1.26.4). CuPy
finds NVRTC from the `nvidia-cuda-nvrtc-cu12` wheel. On NixOS hosts (steve) the driver's libraries are in
`/run/opengl-driver/lib`, which must be on `LD_LIBRARY_PATH`. `ECCENTRIC_GPU_MEMORY` (bytes, default 16 GiB)
bounds the device memory a decoder's workspaces take. Without a GPU, ECCentric imports and runs as before; only
the `_gpu` names fail (`decode()` logs the error and returns None). `tests/test_gpu_decoders.py` skips without a
GPU.

Tested on steve's NVIDIA H200 NVL (driver 595.71). The kernels compute in double precision (needed to match the
CPU bit for bit); GPUs with little double-precision throughput (A40, consumer cards) will be much slower.

## Verification

`analysis/gpu_verification/` (run on steve, 17–18 Nov 2026): seven stored sweep circuits (SD6 noise), 10^6 shots
each sampled once with a fixed seed and saved, decoded by the GPU decoder (all shots) and by the CPU decoder
(chunks of ~10 min on 48 cores; all 10^6 shots where affordable, otherwise the first 5×10^4–2×10^5). Per shot we
compare the predicted observable flips. For Relay-BP, `relay_exact.py` also compares error vectors with relay-bp given the same
explicit memory strengths (3,000 shots each of surface d5, bb72, hgp_hamming and color d5: identical). "z" is the paired test on the shots only one of the two decoders gets
wrong, (b − c) / √(b + c); |z| > 3 means the logical error rates differ. Speeds: GPU shots/s over all shots
(decoding only, after building the decoder), CPU shots/s of one core, measured with 48 workers running at once
(10–20% slower than one process alone, so the speedups are that much too high).

| decoder | circuit (SD6) | GPU shots | CPU shots | identical | LER GPU | LER CPU | only GPU / only CPU wrong | z | GPU shots/s | CPU shots/s | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bplsd_gpu` | surface d5, p=0.002 | 1,000,000 | 100,000 | 100.00% | 6.330e-03 | 6.330e-03 ± 2.5e-04 | 0 / 0 | 0.00 | 4,661 | 32.2 | 145× |
| `bplsd_gpu` | color d5, p=0.002 | 1,000,000 | 50,000 | 100.00% | 4.378e-02 | 4.378e-02 ± 9.2e-04 | 0 / 0 | 0.00 | 930 | 9.9 | 94× |
| `bplsd_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 3.302e-03 | 3.302e-03 ± 5.7e-05 | 0 / 0 | 0.00 | 91,520 | 751.8 | 122× |
| `bplsd_gpu` | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 7.000e-06 | 7.000e-06 ± 2.6e-06 | 0 / 0 | 0.00 | 10,384 | 292.1 | 36× |
| `bplsd_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 2.257e-01 | 2.257e-01 ± 4.2e-04 | 0 / 0 | 0.00 | 23,634 | 110.9 | 213× |
| `belief_find_gpu` | surface d5, p=0.002 | 1,000,000 | 100,000 | 100.00% | 5.220e-03 | 5.220e-03 ± 2.3e-04 | 0 / 0 | 0.00 | 5,090 | 30.4 | 168× |
| `belief_find_gpu` | color d5, p=0.002 | 1,000,000 | 50,000 | 100.00% | 2.238e-02 | 2.238e-02 ± 6.6e-04 | 0 / 0 | 0.00 | 1,115 | 9.1 | 122× |
| `belief_find_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 3.575e-03 | 3.575e-03 ± 6.0e-05 | 0 / 0 | 0.00 | 104,515 | 742.4 | 141× |
| `belief_find_gpu` | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 1.200e-05 | 1.200e-05 ± 3.5e-06 | 0 / 0 | 0.00 | 10,074 | 259.8 | 39× |
| `belief_find_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 2.107e-01 | 2.107e-01 ± 4.1e-04 | 0 / 0 | 0.00 | 22,295 | 120.4 | 185× |
| `gari_gpu` | surface d5, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 7.600e-03 | 7.600e-03 ± 8.7e-05 | 0 / 0 | 0.00 | 452,526 | 5,995.2 | 75× |
| `gari_gpu` | color d5, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 1.297e-02 | 1.297e-02 ± 1.1e-04 | 0 / 0 | 0.00 | 418,828 | 5,694.7 | 74× |
| `gari_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 3.704e-03 | 3.704e-03 ± 6.1e-05 | 0 / 0 | 0.00 | 69,522 | 3,270.5 | 21× |
| `gari_gpu` | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 4.500e-05 | 4.500e-05 ± 6.7e-06 | 0 / 0 | 0.00 | 34,381 | 531.5 | 65× |
| `gari_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 1.298e-01 | 1.298e-01 ± 3.4e-04 | 0 / 0 | 0.00 | 134,864 | 2,053.2 | 66× |
| `gari_gpu` | steane d9, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 4.598e-02 | 4.598e-02 ± 2.1e-04 | 0 / 0 | 0.00 | 13,727 | 77.6 | 177× |
| `gari_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 3.415e-01 | 3.415e-01 ± 4.7e-04 | 0 / 0 | 0.00 | 68,709 | 942.5 | 73× |
| `relay_bp_gpu` | surface d5, p=0.002 | 1,000,000 | 1,000,000 | 99.87% | 2.365e-03 | 2.329e-03 ± 4.8e-05 | 657 / 621 | 1.01 | 128,870 | 528.3 | 244× |
| `relay_bp_gpu` | color d5, p=0.002 | 1,000,000 | 1,000,000 | 99.18% | 9.964e-03 | 1.011e-02 ± 1.0e-04 | 4,049 / 4,192 | -1.58 | 39,385 | 244.3 | 161× |
| `relay_bp_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 99.84% | 2.995e-03 | 3.007e-03 ± 5.5e-05 | 685 / 697 | -0.32 | 147,477 | 482.2 | 306× |
| `relay_bp_gpu` | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 1.000e-06 | 1.000e-06 ± 1.0e-06 | 0 / 0 | 0.00 | 9,347 | 95.2 | 98× |
| `relay_bp_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 95.91% | 9.238e-02 | 9.265e-02 ± 2.9e-04 | 16,430 / 16,698 | -1.47 | 997,675 | 2,208.2 | 452× |
| `relay_bp_gpu` | steane d9, p=0.001 | 1,000,000 | 50,000 | 98.98% | 1.076e-02 | 1.072e-02 ± 4.6e-04 | 255 / 253 | 0.09 | 1,064 | 12.4 | 86× |
| `relay_bp_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 84.18% | 2.918e-01 | 2.918e-01 ± 4.5e-04 | 29,331 / 29,355 | -0.10 | 284,680 | 643.3 | 443× |
| `hier_bposd_hard_gpu` | steane d9, p=0.001 | 50,000 | 50,000 | 100.00% | 5.218e-02 | 5.218e-02 ± 9.9e-04 | 0 / 0 | 0.00 | 9 | 5.4 | 2× |
| `hier_bposd_hard_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 4.602e-01 | 4.602e-01 ± 5.0e-04 | 0 / 0 | 0.00 | 965 | 261.2 | 4× |
| `hier_bposd_gpu` | steane d9, p=0.001 | 1,000,000 | 50,000 | 99.39% | 1.504e-02 | 1.508e-02 ± 5.5e-04 | 152 / 154 | -0.11 | 18 | 8.7 | 2× |
| `hier_bposd_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 99.01% | 3.104e-01 | 3.104e-01 ± 4.6e-04 | 1,066 / 1,012 | 1.18 | 3,413 | 381.0 | 9× |
| `bposd_gpu` | surface d5, p=0.002 | 1,000,000 | 100,000 | 99.84% | 3.490e-03 | 3.560e-03 ± 1.9e-04 | 78 / 85 | -0.55 | 4,488 | 30.3 | 148× |
| `bposd_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 99.89% | 3.104e-03 | 3.124e-03 ± 5.6e-05 | 370 / 390 | -0.73 | 18,459 | 739.0 | 25× |
| `bposd_gpu` | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 0.000e+00 | 1.000e-06 ± 1.0e-06 | 0 / 1 | -1.00 | 3,234 | 264.0 | 12× |
| `bposd_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 98.41% | 1.999e-01 | 1.998e-01 ± 4.0e-04 | 3,562 / 3,427 | 1.61 | 8,149 | 110.9 | 73× |
| `bposd_tuned_gpu` | surface d5, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 2.394e-03 | 2.396e-03 ± 4.9e-05 | 3 / 5 | -0.71 | 26,556 | 340.1 | 78× |
| `bposd_tuned_gpu` | color d5, p=0.002 | 1,000,000 | 1,000,000 | 100.00% | 1.421e-02 | 1.421e-02 ± 1.2e-04 | 22 / 25 | -0.44 | 21,335 | 200.2 | 107× |
| `bposd_tuned_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 99.98% | 5.088e-03 | 5.102e-03 ± 7.1e-05 | 59 / 73 | -1.22 | 20,715 | 234.5 | 88× |
| `bposd_tuned_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 99.98% | 1.417e-01 | 1.417e-01 ± 3.5e-04 | 22 / 15 | 1.15 | 56,171 | 2,117.0 | 27× |
| `bposd_tuned_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 98.10% | 2.855e-01 | 2.855e-01 ± 4.5e-04 | 3,760 / 3,731 | 0.34 | 52,562 | 1,490.1 | 35× |
| `union_find_gpu` | surface d5, p=0.002 | 1,000,000 | 1,000,000 | 99.71% | 1.258e-02 | 1.262e-02 ± 1.1e-04 | 1,450 / 1,488 | -0.70 | 1,019,321 | 3,865.2 | 264× |
| `union_find_gpu` | color d5, p=0.002 | 1,000,000 | 1,000,000 | 98.97% | 4.301e-02 | 4.220e-02 ± 2.0e-04 | 5,580 / 4,763 | 8.03 | 768,395 | 2,786.3 | 276× |
| `union_find_gpu` | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 53.52% | 4.655e-01 | 4.668e-01 ± 5.0e-04 | 32,649 / 33,959 | -5.08 | 238,240 | 517.7 | 460× |
| `union_find_gpu` | gross d12, p=0.001 | 1,000,000 | 200,000 | 67.31% | 3.228e-01 | 3.231e-01 ± 1.0e-03 | 1,807 / 1,864 | -0.94 | 23,387 | 47.9 | 489× |
| `union_find_gpu` | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 88.02% | 3.830e-01 | 3.840e-01 ± 4.9e-04 | 31,637 / 32,599 | -3.80 | 2,023,043 | 9,388.4 | 215× |
| `union_find_gpu` | steane d9, p=0.001 | 1,000,000 | 1,000,000 | 95.14% | 1.200e-01 | 1.221e-01 ± 3.3e-04 | 23,252 / 25,301 | -9.30 | 86,390 | 405.0 | 213× |
| `union_find_gpu` | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 72.56% | 4.851e-01 | 4.847e-01 ± 5.0e-04 | 14,122 / 13,679 | 2.66 | 1,436,157 | 6,921.2 | 208× |
| nv-qldpc Relay-BP (replaced) | surface d5, p=0.002 | 1,000,000 | 1,000,000 | 99.88% | 2.347e-03 | 2.329e-03 ± 4.8e-05 | 632 / 614 | 0.51 | 31,557 | 528.3 | 60× |
| nv-qldpc Relay-BP (replaced) | color d5, p=0.002 | 1,000,000 | 1,000,000 | 99.18% | 1.006e-02 | 1.011e-02 ± 1.0e-04 | 4,089 / 4,134 | -0.50 | 13,996 | 244.3 | 57× |
| nv-qldpc Relay-BP (replaced) | bb72 d6, p=0.002 | 1,000,000 | 1,000,000 | 99.84% | 3.091e-03 | 3.007e-03 ± 5.5e-05 | 749 / 665 | 2.23 | 16,722 | 482.2 | 35× |
| nv-qldpc Relay-BP (replaced) | gross d12, p=0.001 | 1,000,000 | 1,000,000 | 100.00% | 0.000e+00 | 1.000e-06 ± 1.0e-06 | 0 / 1 | -1.00 | 4,178 | 95.2 | 44× |
| nv-qldpc Relay-BP (replaced) | hgp_hamming d3, p=0.001 | 1,000,000 | 1,000,000 | 95.53% | 9.656e-02 | 9.265e-02 ± 2.9e-04 | 20,146 / 16,231 | 20.53 | 77,123 | 2,208.2 | 35× |
| nv-qldpc Relay-BP (replaced) | steane d9, p=0.001 | 1,000,000 | 50,000 | 98.96% | 1.058e-02 | 1.072e-02 ± 4.6e-04 | 257 / 264 | -0.31 | 1,511 | 12.4 | 122× |
| nv-qldpc Relay-BP (replaced) | mhc36 d4, p=0.001 | 1,000,000 | 1,000,000 | 84.16% | 2.919e-01 | 2.918e-01 ± 4.5e-04 | 29,253 / 29,185 | 0.28 | 47,901 | 643.3 | 74× |

### Known differences

- **`union_find_gpu`: ties.** Union-find with inversion orders a cluster's bits by weight and solves on the first
  independent ones, so its correction depends on how bits of equal weight are ordered — and an error model has
  few distinct priors (surface d5: 86 distinct weights over 1,677 errors). ldpc orders ties by the iteration order
  of its hash sets (`tsl::robin_set`) followed by an unstable `std::sort`, an implementation detail. The logical
  error rate depends on it: on the GPU, ordering ties by index, by reverse index or randomly moves it from 0.668 to
  0.717 on hgp_hamming d3 and from 0.0417 to 0.0441 on color d5 (2×10^5 shots, ±0.001 and ±0.0005). The GPU
  version orders ties by a fixed random permutation, which behaves like ldpc's order on average but is not the same
  order: the logical error rates differ by 0.1–2% (relative) and the paired test detects it on 5 of 7 circuits
  (color d5 4.30% vs 4.22%, steane d9 12.0% vs 12.2%, in both directions). Treat `union_find_gpu` as union-find
  with a different tie order, not as a copy of `union_find`; reproducing ldpc's order would mean emulating its hash
  sets and sort on the GPU.
- **nv-qldpc's Relay-BP uses one set of memory strengths for all shots** (why `relay_bp_gpu` no longer uses it).
  It was 4% worse than relay-bp on hgp_hamming d3 (0.0966 vs 0.0927, z = 20) and slightly worse on bb72
  (z = 2.2). The message clip, `repeatable` and fp32/fp64 change nothing (2×10^5 shots: 0.0971–0.0972), but the
  seed does: seeds 1–7 and 42 give 0.0891–0.1023, far beyond the statistical error (±0.0007). relay-bp draws new
  memory strengths for every leg of every shot, so its logical error rate averages over the draws; nv-qldpc
  evidently draws them from the seed alone, so each seed is a different fixed decoder. Our kernel draws them per
  shot and agrees with relay-bp on all seven circuits (|z| ≤ 1.6; hgp_hamming 0.0924 vs 0.0927).
- **nv-qldpc is closed source**; its BP-OSD agrees with ldpc statistically, not shot by shot, and without `repeatable`
  two runs on the same shots can differ.

### The CPU `hier_bposd` reuses the previous shot's level posteriors

ldpc's `BpDecoder.decode` returns at once for a zero syndrome without running BP and without touching its
`log_prob_ratios`. `hierarchical.Hierarchical` reads those ratios after every level, so for a level whose
detectors are all zero it passes on the posteriors of the last earlier shot (in sampling order, across batches)
that had a nonzero syndrome at that level — another shot's information. `hier_bposd_gpu` reproduces this by
default (`zero_syndrome="previous"`), which is why it matches the CPU decoder; `zero_syndrome="run"` runs BP on
the zero syndrome instead, as the method intends. How much this changes the merged `hier_bposd` results has not
been measured yet; fixing it would change them.

### Speed

Speedups over one CPU core: `union_find` 208–489×, `bplsd`/`belief_find` 36–213×, `gari` 21–177×, `bposd`
12–148×, `relay_bp` 86–452×, `bposd_tuned` 27–107×, `hier_bposd` 2–9×. The fastest reach 0.4–2 million shots/s
on small codes. Where BP fails on many shots, the CPU post-processing dominates: `hier_bposd_gpu` on steane d9
sent 44% of the shots to OSD on the CPU and ran at 18 shots/s (10^6 shots took 15.6 h); `bplsd_gpu` on color d5
sent 23%. The speed of a GPU decoder therefore depends on the circuit and the noise in a different way than on the
CPU, and one speedup factor per decoder does not describe it.

### How well a short pilot predicts a GPU decoder's speed

For time estimates (e.g. in Matchmaker), `analysis/gpu_verification/pilot_timing.py` measured, for each of the 42
verified (circuit, decoder) pairs except the two steane d9 `hier_bposd` ones: the decoder's build time, Stim's
sampling time, a reference run (all 10^6 shots, or as many as take ~5 min; run twice) and pilots of 10^3, 10^4
and 10^5 shots at three offsets into the shots, all in `decode()`'s batch sizes (results in
`analysis/gpu_verification/pilot/`). The Relay-BP pairs were measured with nv-qldpc's Relay-BP, before `relay_bp_gpu` moved to
our kernel.

| pilot | within 20% of the reference throughput | within 2× | worst |
|---|---:|---:|---:|
| 10^3 shots | 16% | 56% | 24× |
| 10^4 shots | 72% | 90% | 4.3× |
| 10^5 shots | 98% | 100% | 1.49× |

Small pilots do not fill the GPU and underestimate the throughput; a 10^5-shot pilot (seconds for most pairs, up
to ~2.5 min for the slowest hybrid decoders) predicts it within 20% in all but one case. That case, `bplsd_gpu` on
color d5, sends 23% of its shots to LSD on the CPU, and its two reference runs differed by 56% (other load on the
host); elsewhere repeated runs differ by under 2% (median 0.9%). Building a decoder takes 2–9 s (median 3.9 s).
Sampling with Stim costs a median 5% of the decoding time; only union-find on the smallest codes decodes as fast
as Stim samples.
