# Circuits from outside ECCentric

Matchmaker's packaged model (trained on ECCentric only) on Stim's generated memory circuits with uniform circuit noise and on decoder-bench's circuits with SI1000 noise, decoded and timed by ECCentric's sweep worker like the training data (target 100 errors, at most 200,000 shots or 10 minutes). The model is given each circuit's code distance. Errors are in decades (|log10 error|); intervals are the model's conformalized [q10, q90]. Trees alone, without the scaling laws: see EXTRAPOLATION.md.

| set | circuits | regime correct | best-LER median error | p90 | in [q10, q90] | time within 3x | choice good enough (accurate / cheapest) | always BP-OSD | always Tesseract |
|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 100% | 0.05 | 0.28 | 86% | 87% | 100% / 94% | 67% | 100% |
| stim | 27 | 100% | 0.04 | 0.30 | 83% | 88% | 100% / 92% | 62% | 100% |
| decoder-bench | 7 | 100% | 0.13 | 0.20 | 100% | 84% | 100% / 100% | 80% | 100% |

Time per shot within 3x, per decoder (all circuits): mwpm 87% (23), bposd 82% (34), bplsd 85% (34), belief_find 88% (34), relay_bp 88% (34), tesseract 91% (34).

| circuit | detectors | best measured | predicted best LER [q10, q90] | P(above) | recommended (accurate / cheapest) | outside training |
|---|---|---|---|---|---|---|
| decoder_bench_bb_d12_p0.003 | 936 | tesseract 0.0017 | 0.04 [0.01, 0.085] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_bb_d6_p0.003 | 252 | tesseract 0.074 | 0.12 [0.07, 0.22] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_bb_d6_p0.005 | 252 | tesseract 0.42 | 0.47 [0.28, 0.68] | 0.04 | tesseract / relay_bp |  |
| decoder_bench_color_d7_p0.003 | 252 | tesseract 0.0057 | 0.011 [0.0063, 0.023] | 0.00 | tesseract / tesseract |  |
| decoder_bench_color_d7_p0.005 | 252 | tesseract 0.03 | 0.074 [0.04, 0.14] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_surface_d7_p0.003 | 336 | tesseract 0.0038 | 0.004 [0.0019, 0.013] | 0.00 | tesseract / tesseract |  |
| decoder_bench_surface_d7_p0.005 | 336 | tesseract 0.027 | 0.028 [0.013, 0.062] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.001 | 9 | tesseract 0.0031 | 0.0025 [0.0013, 0.0061] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.003 | 9 | tesseract 0.015 | 0.014 [0.0093, 0.029] | 0.00 | tesseract / relay_bp |  |
| stim_color_d5_p0.001 | 45 | tesseract 0.00052 | 0.0005 [0.00032, 0.0017] | 0.00 | tesseract / tesseract |  |
| stim_color_d5_p0.003 | 45 | tesseract 0.0096 | 0.01 [0.0058, 0.024] | 0.00 | tesseract / relay_bp |  |
| stim_color_d7_p0.001 | 126 | tesseract 0.00011 | 0.00011 [6.9e-05, 0.0004] | 0.00 | tesseract / tesseract |  |
| stim_color_d7_p0.003 | 126 | tesseract 0.0039 | 0.0066 [0.0036, 0.013] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.001 | 270 | relay_bp 5.9e-05 | 2.7e-05 [1.6e-05, 7.6e-05] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.003 | 270 | tesseract 0.0014 | 0.0034 [0.002, 0.0063] | 0.00 | tesseract / tesseract |  |
| stim_repetition_d11_p0.01 | 120 | mwpm 1.5e-05 | 0.0026 [0.0013, 0.006] | 0.00 | tesseract / mwpm |  |
| stim_repetition_d21_p0.01 | 440 | mwpm 1.9e-05 (bound) | 2.1e-05 [1.1e-05, 6.8e-05] | 0.00 | tesseract / relay_bp |  |
| stim_repetition_d5_p0.01 | 24 | bposd 0.0015 | 0.0094 [0.004, 0.028] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d11_p0.003 | 1320 | mwpm 0.00021 | 0.00022 [0.00014, 0.00074] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d11_p0.007 | 1320 | tesseract 0.022 (bound) | 0.017 [0.0082, 0.046] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d3_p0.001 | 24 | tesseract 0.00068 | 0.00063 [0.00043, 0.0014] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d3_p0.003 | 24 | tesseract 0.0061 | 0.0055 [0.0037, 0.0094] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d3_p0.007 | 24 | tesseract 0.031 | 0.03 [0.017, 0.047] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d5_p0.001 | 120 | tesseract 7.5e-05 | 0.00011 [6.2e-05, 0.00043] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d5_p0.003 | 120 | tesseract 0.0019 | 0.0019 [0.0013, 0.0053] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d5_p0.007 | 120 | tesseract 0.024 | 0.019 [0.012, 0.043] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d7_p0.001 | 336 | tesseract 1.9e-05 (bound) | 2e-05 [7.1e-06, 0.00013] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.003 | 336 | tesseract 0.00024 | 0.00077 [0.00028, 0.0027] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.007 | 336 | tesseract 0.018 | 0.017 [0.0083, 0.036] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.003 | 720 | tesseract 0.00025 (bound) | 0.00052 [0.00014, 0.0018] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.007 | 720 | tesseract 0.012 | 0.018 [0.0071, 0.043] | 0.00 | tesseract / relay_bp |  |
| stim_surface_unrotated_d3_p0.003 | 36 | tesseract 0.0059 | 0.0082 [0.005, 0.019] | 0.00 | tesseract / relay_bp |  |
| stim_surface_unrotated_d5_p0.003 | 200 | tesseract 0.0014 | 0.003 [0.0016, 0.0098] | 0.00 | tesseract / tesseract |  |
| stim_surface_unrotated_d7_p0.003 | 588 | tesseract 0.00016 | 0.00099 [0.00039, 0.0046] | 0.00 | tesseract / tesseract |  |
