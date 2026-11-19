# Codes

## Codes of the SIGMETRICS'26 paper

| Name | Code |
| --- | --- |
| `surface` | Rotated surface code |
| `hh` | Heavy-hex code |
| `gross` | Bivariate bicycle ("gross") codes, e.g. [[144,12,12]] |
| `color` | Triangular color code (6.6.6) |
| `bacon` | Bacon-Shor code |
| `steane` | Concatenated Steane code, d = 3, 9, 27 |

## Code catalog

Besides the codes of the paper, these codes are available. They are registered in `eccentric/codes/catalog.py`, which builds a noiseless Stim memory circuit (Z basis) for each of them; the circuit then goes through the same transpilation, noise and decoding pipeline as every other code. Code families accept any distance from their minimum upwards; fixed instances only have their own distance, so use `backends_sizes` (or `distances` with exactly that value) for them. The qubit count includes the syndrome ancillas.

| Name | Code | Qubits | Source |
| --- | --- | --- | --- |
| `repetition` | Repetition code, distance `d ≥ 3` | 2d − 1 | Stim generator |
| `xzzx` | XZZX surface code (rotated surface code with a Hadamard on every other data qubit), `d ≥ 3` | 2d² − 1 | Stim generator |
| `toric` | Unrotated toric code [[2d², 2, d]] | 4d² | [qLDPC](https://github.com/qLDPCOrg/qLDPC) |
| `honeycomb` | Periodic honeycomb (Floquet) code, SD6 circuit, even `d ≥ 4` | 2.5 · d · 6⌈d/4⌉ | [honeycomb_threshold](https://github.com/Strilanc/honeycomb_threshold) |
| `planar_honeycomb` | Planar honeycomb code with boundaries, SD6 circuit, `d ≥ 3` | 68 at d = 3 | [honeycomb-boundaries](https://github.com/Strilanc/honeycomb-boundaries) |
| `bb54`, `bb72`, `bb90`, `bb98`, `bb108`, `bb162`, `bb180`, `bb288`, `bb756` | Bivariate bicycle codes [[54,8,6]], [[72,12,6]], [[90,8,10]], [[98,6,12]], [[108,8,10]], [[162,8,14]], [[180,8,16]], [[288,12,18]], [[756,16,≤34]], same circuit as `gross` | 2n | Bravyi et al., arXiv:2308.07915, and `BBCODE_DICT` |
| `bpc36`, `bpc54`, `bpc72`, `bpc90`, `bpc108`, `bpc126`, `bpc144`, `bpc162`, `bpc180` | Balanced product cyclic codes [[36,8,4]], [[54,8,4]], [[72,8,8]], [[90,8,10]], [[108,8,8]], [[126,8,10]], [[144,8,12]], [[162,8,12]], [[180,8,16]], cardinal schedule | 2n | [QUITS](https://github.com/mkangquantum/quits), arXiv:2411.03302 |
| `hgp225`, `hgp625`, `hgp900` | Hypergraph products of classical (3,4)-regular LDPC codes, [[225,9,6]], [[625,25,8]], [[900,36,10]], cardinal schedule | 441, 1225, 1764 | QUITS (matrices in `eccentric/codes/data/quits_hgp`) |
| `lcs15`, `lcs52`, `lcs65`, `lcs150`, `lcs175` | Lift-connected surface codes [[15,3,3]], [[52,4,4]], [[65,5,5]], [[150,6,6]], [[175,7,7]], zxcoloration schedule | 27, 100, 125, 294, 343 | QUITS, arXiv:2401.02911 |
| `qlp544` | Quasi-cyclic lifted product code [[544,80,≤12]], cardinal schedule | 1024 | QUITS, arXiv:2308.08648 |
| `mitten150` | Mitten code [[150,30,≤8]] over C5 × S3 (QUITS' demo instance, not the d = 10 code of the paper), Mitten schedule | 270 | QUITS, arXiv:2607.28795 |
| `gb46`, `gb48`, `gb126` | Generalized bicycle codes [[46,2,9]], [[48,6,8]], [[126,28,8]] | 2n | Panteleev & Kalachev, arXiv:1904.02703; circuit from qLDPC |
| `hgp_hamming` | Hypergraph product of two [7,4,3] Hamming codes, [[58,16,3]] | 100 | qLDPC |
| `rm15` | Punctured quantum Reed-Muller code [[15,1,3]] | 29 | qLDPC |
| `lp` | Lifted product (lift-connected surface) code [[39,3,3]] | 75 | qLDPC |
| `qt36`, `qt72` | Quantum Tanner codes [[36,8,3]], [[72,14,4]] on the dihedral group D4 | 68, 136 | qLDPC |
| `cbb30`, `cbb42`, `cbb70`, `cbb108`, `cbb126`, `cbb154` | Coprime bivariate bicycle codes [[30,4,6]], [[42,6,6]], [[70,6,8]], [[108,12,6]], [[126,12,10]], [[154,6,16]]: bivariate bicycle codes with coprime orders l, m and polynomials in π = xy, weight 6 | 2n | Wang & Mueller, arXiv:2408.10001 (Table 2); circuit from qLDPC |
| `tb30`, `tb48`, `tb72`, `tb144` | Trivariate bicycle codes [[30,4,5]], [[48,6,6]], [[72,4,8]], [[144,2,12]]: bicycle codes whose polynomials also use z = xy, check weights 5, 6, 5 and 4 | 2n | Voss et al., arXiv:2406.19151 (Appendix A); circuit from qLDPC |
| `mhc36`, `mhc216` | Many-hypercube codes [[36,16,4]], [[216,64,8]]: [[6,4,2]] concatenated 2 and 3 times, checks of weight 6–24 | 56, 368 | Goto, arXiv:2403.16054; qLDPC |
| `golay23` | Quantum Golay code [[23,1,7]] from the classical [23,12,7] Golay code, weight-8 checks | 45 | qLDPC |
| `hyperbolic_floquet_60`, `_120`, `_160`, `_240`, `_256`, `_320` | Hyperbolic Floquet codes [[60,5,3]], [[120,8,3]], [[160,10,4]], [[240,8,4]], [[256,34,4]], [[320,18,5]] (pair measurements, no ancillas) | n | [hyperbolic-floquet-data](https://github.com/oscarhiggott/hyperbolic-floquet-data) |
| `semi_hyperbolic_floquet_64`, `_144`, `_256`, `_400`, `_576` | Semi-hyperbolic Floquet codes from the Bolza code, k = 4, d = 3/4/6/7/8 (a distance-scaling family) | n | hyperbolic-floquet-data |

Syndrome-extraction schedules (`schedules`), the first being each code's default:

| Codes | Schedules |
| --- | --- |
| `bb*` | `ibm` (the depth-8 circuit of Bravyi et al., as for `gross`), `zxcoloration` (QUITS) |
| `toric`, `hgp_hamming`, `rm15`, `lp`, `gb*`, `qt*`, `cbb*`, `tb*`, `mhc*`, `golay23` | `edge_coloring` (all checks together), `edge_coloring_xz` (X checks before Z checks) (qLDPC) |
| `bpc*`, `hgp225`/`625`/`900`, `qlp544` | `cardinal`, `cardinalNSmerge`, `zxcoloration` (QUITS) |
| `lcs*` | `zxcoloration`, `cardinal`, `cardinalNSmerge` (QUITS) |
| `mitten150` | `custom` (depth 12), `hook_error_free` (depth 24), `zxcoloration` (QUITS) |

The other codes have a single circuit. Result rows report the schedule used.

## Notes

- Decoders: `bposd` works for all of them. `mwpm` works for `repetition`, `xzzx`, `toric`, `honeycomb`, `planar_honeycomb` and the (semi-)hyperbolic Floquet codes; the other codes are not matchable. `chromobius` needs color-annotated detector coordinates, which only the paper's `color` code has.
- `d` is the code distance. The syndrome-extraction circuits can have a lower circuit-level distance: `lp` (qLDPC's edge-coloring schedule), the `lcs*` codes and the honeycomb codes have circuit distance below `d`, and `bpc72` below 8.
- `cycles` sets the number of syndrome rounds. Floquet codes count one round as one X, Y and Z sub-round; the planar honeycomb code needs an even number of at least 4 rounds and the hyperbolic Floquet codes run `2 + 2⌈(cycles − 2)/2⌉` rounds.
- Distances verified by exact computation: `bb54`, `bb98`, `gb46`, `gb48`, `rm15`, `hgp_hamming`, `lp`, `lcs15`, `lcs52`, `qt36`, `qt72`, `cbb30`, `cbb42`, `cbb70`, `tb30`, `tb48`, `tb72`, `mhc36`, `golay23`, and `toric` at d = 3. The `hgp*`, other `lcs*`, `qlp544`, `mitten150`, `cbb108`, `cbb126`, `cbb154`, `tb144` and `mhc216` distances are upper bounds from QUITS' randomized estimator, which match the cited papers. The others are taken from the cited papers; for `bb756` the distance is an upper bound.
- `hgp625`, `hgp900` and `qlp544` take up to a minute to build; their tests only run with `pytest -m slow`.

## Lattice surgery

Lattice-surgery operations on rotated surface-code patches are available as codes, built with [tqec](https://github.com/tqec/tqec) (`eccentric/codes/lattice_surgery.py`): each is a tqec block graph compiled into a noiseless Stim circuit with tqec's detectors and observables, which then goes through ECCentric's pipeline like any memory circuit (devices, routing, noise models, decoders). Patches have odd distance `d ≥ 3` and every block lasts `d` syndrome rounds, so `cycles` must be left unset (or equal `d`). Rows report a failure when any of the operation's observables is wrong.

| Name | Operation | Qubits | Rounds |
| --- | --- | --- | --- |
| `ls_memory` | One patch, prepared and read out in the basis (tqec's memory) | 2d² − 1 | d |
| `ls_merge` | Two patches merged for one block and split again: a ZZ measurement in the Z basis, XX in the X basis; the observable is both patches' joint logical operator, which the merge must preserve | 4d² + 2d − 1 | 3d |
| `ls_cnot` | The lattice-surgery CNOT of tqec's gallery (control, target and ancilla patch); its observables are the Z (Z basis) or X (X basis) correlations a CNOT maps | 6d² + 4d − 1 | 4d |

[analysis/LATTICE_SURGERY.md](../analysis/LATTICE_SURGERY.md) checks the memory and the merge against lattice-sim (Maurya and Tannu, ISCA'25) under the same noise.

## Fault-tolerant operations on CSS codes

The `operations` field runs a fault-tolerant operation on a code instead of its memory (`eccentric/codes/operations.py`, built with the [qLDPC](https://github.com/qLDPCOrg/qLDPC) library, whose surgery subpackage its authors mark experimental). Every operation is a noiseless Stim circuit with detectors and observables that goes through the pipeline like a memory circuit; a shot fails if any observable is wrong. All use qLDPC's syndrome extraction (edge coloring, one ancilla per check), so `memory_ec` is their baseline; the catalog's memory circuits may use another schedule. They need `distances`; `cycles` defaults to the distance.

| Operation | What it does | Basis |
| --- | --- | --- |
| `memory_ec` | The memory experiment with qLDPC's syndrome extraction | Z or X |
| `measure_z`, `measure_x` | Fault-tolerant measurement of the first logical qubit's Z (X) by code surgery: a gadget of ancillas attached to the logical operator's support for `cycles` rounds, then detached (Webster, Smith, Cohen, arXiv:2511.15989; circuit per Cain et al., arXiv:2603.28627); gadgets are augmented to a boundary Cheeger constant of 1 where qLDPC can (Williamson and Yoder, Nature Physics 2026) | Z (X) |
| `joint_zz`, `joint_xx` | Joint ZZ (XX) measurement of the first logical qubits of two blocks, the gadgets joined by a universal adapter (Swaroop, Jochym-O'Connor, Yoder, arXiv:2410.03628) | Z (X) |
| `transversal_cnot` | Two blocks, `cycles` rounds each, one layer of CX between corresponding data qubits (a logical CNOT between every pair of logical qubits), `cycles` more rounds; decoders see the errors that cross the gate as hyperedges between the blocks | Z or X |

Available for the catalog's CSS codes (bicycle, balanced product, hypergraph and lifted product, lift-connected surface, quantum Tanner, many-hypercube, Golay, Reed-Muller, toric), the surface code, the gross code and the Steane code at d = 3. The circuit-level distance is that of qLDPC's syndrome extraction (the Steane code's drops to 2 from hook errors; the surface code keeps its distance). Result rows report the `operation` (`memory` for a memory experiment).

```yaml
experiments:
  - name: "Operations"
    num_samples: 100000
    backends: ["custom_full"]
    codes: ["bb72", "hgp_hamming"]
    distances: [6, 3]
    error_types: ["sd6"]
    error_probabilities: [0.001]
    decoders: ["bposd"]
    operations: ["memory_ec", "measure_z", "transversal_cnot"]
```
