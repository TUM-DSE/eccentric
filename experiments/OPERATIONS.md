# Logical operations: predicted from memory experiments

ECCentric's sweep now has logical operations besides memory (ECCentric deeb90c):

- **fault_tolerant_ops**: 13 CSS codes (bivariate bicycle, coprime and trivariate bicycle, balanced product,
  lift-connected surface, hypergraph product, many-hypercube, Golay, Steane, surface, toric, ...) built with the
  qLDPC library: memory with qLDPC's syndrome extraction (`memory_ec`), logical Z and X measurements by surgery
  (`measure_z`, `measure_x`), joint ZZ and XX measurements by surgery (`joint_zz`, `joint_xx`) and a transversal
  CNOT between two blocks (`transversal_cnot`), under SD6 at 1e-3 and 2e-3 and on Harvard, Infleqtion, Apollo
  and Flamingo; decoded with BP-OSD, Tesseract and Relay-BP (and MWPM on surface and toric codes).
- **lattice_surgery**: rotated surface-code patches built with tqec: memory (`ls_memory`), a ZZ/XX merge
  (`ls_merge`) and a CNOT (`ls_cnot`) at d = 3, 5, 7, both bases, under SD6 and on real devices.

The `operations` split (`qec-matchmaker evaluate --group-by operations`) holds out all 580 operation circuits
(1,984 decodes) and predicts them from the 5,044 memory circuits, with no operation circuit in training.
Full report: [operations/REPORT_operations.md](operations/REPORT_operations.md). The baseline is the standard
evaluation on the same table ([reports/REPORT.md](../reports/REPORT.md): 10 folds over 70 held-out codes).
Errors are in decades (|log10 error|).

| | regime correct (recall) | best LER median / p90 error | in [q10, q90] | time within 3x | choice good enough (accurate / cheapest) |
|---|---|---|---|---|---|
| held-out codes (baseline) | 97% (91%) | 0.10 / 0.41 | 83% | 94% | 92% / 90% |
| operations, from memory only | 96% (64%) | 0.15 / 0.60 | 74% | 92% | 88% / 86% |

Per operation (worst LER first):

| operation | circuits | above threshold | regime correct | best LER median / p90 error | time within 3x | choice good enough (accurate) |
|---|---|---|---|---|---|---|
| measure_x | 77 | 1% | 99% | 0.48 / 0.75 | 84% | 77% |
| measure_z | 77 | 7% | 93% | 0.46 / 0.82 | 90% | 74% |
| joint_xx | 71 | 7% | 93% | 0.35 / 0.48 | 93% | 84% |
| ls_cnot | 42 | 12% | 93% | 0.34 / 0.65 | 87% | 100% |
| joint_zz | 66 | 2% | 98% | 0.27 / 0.54 | 92% | 71% |
| ls_merge | 46 | 0% | 100% | 0.19 / 0.51 | 96% | 100% |
| transversal_cnot | 75 | 35% | 93% | 0.16 / 0.40 | 92% | 85% |
| ls_memory | 48 | 0% | 100% | 0.09 / 0.20 | 99% | 88% |
| memory_ec | 78 | 17% | 96% | 0.03 / 0.19 | 93% | 94% |

## What this says

- **Decode time carries over.** 92% of operation decodes are within 3x (94% on held-out memory codes);
  Relay-BP 95%, BP-OSD 92%, MWPM 92%, Tesseract 87%. Time depends on the size and noise of the detector error
  model, which the features describe whatever the circuit does.
- **Memory-like circuits are predicted as well as memory.** `memory_ec` (0.03 decades) and `ls_memory` (0.09)
  are memory experiments built by another tool; the features see no difference.
- **The LER of surgery measurements is the furthest from anything in training**: 0.46-0.48 decades median for
  the single-logical measurements, 0.27-0.35 for joint measurements and the lattice-surgery CNOT. The gadget
  (an ancilla system merged with the code for several rounds) changes how errors become logical failures in
  ways the circuit-level features do not capture. The intervals cover 74% instead of 80%.
- **Above-threshold operations are missed more often**: regime recall 64% (91% on held-out codes), mostly
  transversal CNOTs, the only operation with many circuits above threshold (35%).
- **Decoder choice holds up, with a caveat.** The operation sweeps ran three or four decoders, so the
  recommended decoder was often not measured (97 of 264 circuits for the accurate choice) and is scored as
  BP-OSD; no operation circuit has all five BP-type and Tesseract decoders. The choice numbers are a lower
  bound. Matchmaker picks Tesseract on 224 of 264 circuits, and is as good as always running Tesseract (88%)
  at slightly less time (3.75x vs 3.92x the cheapest good-enough decoder).

## Now in training

The packaged model is retrained on everything (34,798 decodes, 5,624 circuits, b282764). Adding the operations
leaves the held-out-code scores essentially unchanged (regime 98% -> 97%, LER 0.10 / 0.40 -> 0.10 / 0.41,
time 94% -> 94%, choice 92% / 90% -> 92% / 90%). Operations of a kind already in training will be predicted
better than in the split above; a new kind of gadget should be expected to behave like the surgery
measurements here.
