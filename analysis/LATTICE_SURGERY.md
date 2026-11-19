# Lattice surgery: ECCentric's tqec operations against lattice-sim (2026-11-13)

ECCentric's lattice-surgery operations (`ls_memory`, `ls_merge`, `ls_cnot`; `eccentric/codes/lattice_surgery.py`) are
tqec block graphs compiled into Stim circuits and passed through ECCentric's pipeline. To check them, the memory and
the ZZ merge were compared with lattice-sim, the simulator of Maurya and Tannu, "Synchronization for
Fault-Tolerant Quantum Computers" (ISCA'25; MIT license; vendored in `synchronization-artifact/` of the
`mcm-latency-eccentric` branch), under the same noise: DEPOLARIZE2(p) after two-qubit gates, an X flip with
probability p before measurements, DEPOLARIZE1(1e-4) after single-qubit gates, no reset or idle error (lattice-sim
with its idle model off). ECCentric's side is compiled onto an all-to-all device; both are decoded with MWPM,
400,000 shots each (`analysis/lattice_surgery_crosscheck.py`; data in
`experiment_results/LatticeSurgery/merge_crosscheck.json`).

| memory | p | d | ECCentric (tqec): rounds, LER, per round | lattice-sim: rounds, LER, per round | per-round ratio |
|---|---|---|---|---|---|
| | 0.002 | 3 | 3, 1.13e-03, 3.76e-04 | 4, 1.71e-03, 4.29e-04 | 0.88 |
| | 0.002 | 5 | 5, 3.20e-04, 6.40e-05 | 6, 4.60e-04, 7.67e-05 | 0.83 |
| | 0.002 | 7 | 7, 6.00e-05, 8.57e-06 | 8, 9.50e-05, 1.19e-05 | 0.72 |
| | 0.004 | 3 | 3, 4.53e-03, 1.51e-03 | 4, 6.76e-03, 1.70e-03 | 0.89 |
| | 0.004 | 5 | 5, 2.39e-03, 4.78e-04 | 6, 3.40e-03, 5.68e-04 | 0.84 |
| | 0.004 | 7 | 7, 1.01e-03, 1.44e-04 | 8, 1.56e-03, 1.95e-04 | 0.74 |

| merge | p | d | ECCentric (tqec): rounds, LER, per round | lattice-sim: rounds, LER, per round | per-round ratio |
|---|---|---|---|---|---|
| | 0.002 | 3 | 9, 8.89e-03, 9.96e-04 | 8, 2.50e-03, 3.13e-04 | 3.18 |
| | 0.002 | 5 | 15, 2.49e-03, 1.66e-04 | 12, 6.10e-04, 5.09e-05 | 3.27 |
| | 0.002 | 7 | 21, 5.25e-04, 2.50e-05 | 16, 1.50e-04, 9.38e-06 | 2.67 |
| | 0.004 | 3 | 9, 3.30e-02, 3.78e-03 | 8, 9.13e-03, 1.15e-03 | 3.29 |
| | 0.004 | 5 | 15, 1.90e-02, 1.29e-03 | 12, 4.77e-03, 3.99e-04 | 3.23 |
| | 0.004 | 7 | 21, 8.44e-03, 4.05e-04 | 16, 2.02e-03, 1.27e-04 | 3.20 |

- **Memory:** ECCentric's patch is within 10–40% of lattice-sim per round at every d and p (0.7–0.9×), the
  difference of the two syndrome-extraction schedules (tqec uses CZ and CX, lattice-sim H and CX). The noise is
  matched.
- **Merge:** the same distance scaling (per round, Λ ≈ 6 from d = 3 to 5 and 5 to 7 in both), with ECCentric a
  steady 2.7–3.3× higher per round. The operations differ: tqec's merge splits the patches again and checks ZZ
  through the split, which adds failure paths, while lattice-sim keeps the merged patch until it is read out. Both
  have the full circuit distance (graph-like distance d after compilation, `tests/test_lattice_surgery.py`).
- lattice-sim's merge has no split (its `__lattice_merge` / `__lattice_split` are stubs) and no CNOT, so `ls_cnot`
  has no counterpart there; its circuits are tqec's, which tqec validates.
