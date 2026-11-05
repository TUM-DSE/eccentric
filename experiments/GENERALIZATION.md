# Generalization: held-out code families, devices and noise models

The standard evaluation holds out codes (10 folds over 54 codes). Here every fold holds out a whole code family
(21 folds), a whole device (25) or a whole noise model (12), so the model predicts circuits of a family,
device or noise model it never saw. Same table (1,810 circuits, 6,673 decodes, ECCentric 1649beb), same models,
intervals conformalized within each fold. Full reports: `generalization/REPORT_{family,backend,error_type}.md`;
produced with `qec-matchmaker evaluate --folds 100 --group-by <group> --jobs 4`.

| held out | folds | regime correct | best LER median / p90 error (decades) | in [q10, q90] | time within 3x | Relay-BP time within 3x | choice good enough (accurate / cheapest) | all five decoders measured: accurate good enough |
|---|---|---|---|---|---|---|---|---|
| code (baseline) | 10 | 95% | 0.09 / 0.50 | 83% | 86% | 65% | 91% / 85% | 92% |
| code family | 21 | 94% | 0.11 / 0.64 | 76% | 83% | 65% | 91% / 87% | 92% |
| device | 25 | 94% | 0.07 / 0.61 | 81% | 88% | 85% | 92% / 89% | 94% |
| noise model | 12 | 93% | 0.09 / 0.64 | 80% | 87% | 82% | 91% / 82% | 93% |

Always BP-OSD is good enough on 75% and always Tesseract on 96% (at 3-4x the time of the cheapest good-enough
decoder) in every split.

## What this says

- **Unseen families, devices and noise models cost little on average.** The median LER error and the decoder
  choice barely move; the tail grows (p90 0.50 -> 0.61-0.64 decades) and the intervals cover less (83% -> 76-81%).
- **Relay-BP's time depends on the code, not the device**: holding out codes or families it is 65% within 3x,
  holding out devices or noise models 82-85%.
- **The failures are extrapolations to regimes nothing else covers:**
  - *Apollo* (held out as device or noise model): best LER off by 1.3-1.5 decades (median). It is the only
    device with errors this low, so without it the model has never seen LERs of 1e-5 and below.
  - *Harvard* (atom loss, heralded): regime correct on only 63% of its circuits (70% of them are above
    threshold); no other noise model has loss.
  - *All-to-all SD6 (the reference slice)*: 0.37-0.39 decades.
  - *Families*: repetition (1.4 decades; a classical code, 12 circuits), quantum Reed-Muller (0.9; 4 circuits),
    toric (0.5), surface (0.3, p90 0.9).
- **Where ECCentric data helps most**: lower-noise devices and noise models, and noise models with loss. The
  noise-scaling sweep (device noise x0.1-1, Harvard included, started 2026-11-05) targets exactly these.
