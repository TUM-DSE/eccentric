# Circuits from outside ECCentric

Matchmaker's packaged model (trained on ECCentric only) on Stim's generated memory circuits with uniform circuit noise and on decoder-bench's circuits with SI1000 noise, decoded and timed by ECCentric's sweep worker like the training data (target 100 errors, at most 200,000 shots or 10 minutes). The model is given each circuit's code distance. Errors are in decades (|log10 error|); intervals are the model's conformalized [q10, q90]. Trees alone, without the scaling laws: see EXTRAPOLATION.md.

| set | circuits | regime correct | best-LER median error | p90 | in [q10, q90] | time within 3x | choice good enough (accurate / cheapest) | always BP-OSD | always Tesseract |
|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 100% | 0.15 | 0.55 | 64% | 69% | 100% / 72% | 67% | 100% |
| stim | 27 | 100% | 0.15 | 0.56 | 67% | 69% | 100% / 69% | 62% | 100% |
| decoder-bench | 7 | 100% | 0.20 | 0.33 | 50% | 70% | 100% / 80% | 80% | 100% |

Time per shot within 3x, per decoder (all circuits): mwpm 91% (23), bposd 76% (34), bplsd 71% (34), belief_find 71% (34), relay_bp 38% (34), tesseract 74% (34).

| circuit | detectors | best measured | predicted best LER [q10, q90] | P(above) | recommended (accurate / cheapest) | outside training |
|---|---|---|---|---|---|---|
| decoder_bench_bb_d12_p0.003 | 936 | tesseract 0.0017 | 0.034 [0.0055, 0.1] | 0.00 | tesseract / tesseract |  |
| decoder_bench_bb_d6_p0.003 | 252 | tesseract 0.074 | 0.17 [0.082, 0.24] | 0.00 | tesseract / tesseract |  |
| decoder_bench_bb_d6_p0.005 | 252 | tesseract 0.42 | 0.47 [0.31, 0.79] | 0.20 | tesseract / bposd |  |
| decoder_bench_color_d7_p0.003 | 252 | tesseract 0.0057 | 0.034 [0.011, 0.049] | 0.00 | tesseract / tesseract |  |
| decoder_bench_color_d7_p0.005 | 252 | tesseract 0.03 | 0.14 [0.062, 0.19] | 0.00 | tesseract / tesseract |  |
| decoder_bench_surface_d7_p0.003 | 336 | tesseract 0.0038 | 0.0057 [0.0035, 0.014] | 0.00 | tesseract / tesseract |  |
| decoder_bench_surface_d7_p0.005 | 336 | tesseract 0.027 | 0.037 [0.017, 0.081] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.001 | 9 | tesseract 0.0031 | 0.0029 [0.0008, 0.017] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.003 | 9 | tesseract 0.015 | 0.018 [0.0073, 0.042] | 0.00 | tesseract / mwpm |  |
| stim_color_d5_p0.001 | 45 | tesseract 0.00052 | 0.0013 [0.00024, 0.0024] | 0.00 | tesseract / tesseract |  |
| stim_color_d5_p0.003 | 45 | tesseract 0.0096 | 0.014 [0.0062, 0.025] | 0.00 | tesseract / tesseract |  |
| stim_color_d7_p0.001 | 126 | tesseract 0.00011 | 0.00039 [4.8e-05, 0.0011] | 0.00 | tesseract / tesseract |  |
| stim_color_d7_p0.003 | 126 | tesseract 0.0039 | 0.011 [0.0043, 0.024] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.001 | 270 | tesseract 0.00016 (bound) | 7.4e-05 [8.8e-06, 0.00019] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.003 | 270 | tesseract 0.0014 | 0.0073 [0.0025, 0.01] | 0.00 | tesseract / tesseract |  |
| stim_repetition_d11_p0.01 | 120 | mwpm 1.5e-05 | 0.01 [0.0034, 0.017] | 0.00 | tesseract / mwpm |  |
| stim_repetition_d21_p0.01 | 440 | mwpm 1.9e-05 (bound) | 3.1e-05 [2.2e-05, 0.0001] | 0.00 | tesseract / tesseract |  |
| stim_repetition_d5_p0.01 | 24 | bposd 0.0015 | 0.03 [0.005, 0.055] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d11_p0.003 | 1320 | mwpm 0.00021 | 0.00014 [7.4e-05, 0.00058] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d11_p0.007 | 1320 | tesseract 0.022 (bound) | 0.016 [0.0061, 0.045] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d3_p0.001 | 24 | tesseract 0.00068 | 0.00092 [0.00059, 0.0043] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d3_p0.003 | 24 | tesseract 0.0061 | 0.0081 [0.0048, 0.014] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d3_p0.007 | 24 | tesseract 0.031 | 0.03 [0.017, 0.056] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d5_p0.001 | 120 | tesseract 7.5e-05 | 0.00022 [9.4e-05, 0.00066] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d5_p0.003 | 120 | tesseract 0.0019 | 0.0034 [0.0023, 0.0068] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d5_p0.007 | 120 | tesseract 0.024 | 0.025 [0.018, 0.051] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d7_p0.001 | 336 | relay_bp 1.9e-05 | 3.3e-05 [9.1e-06, 0.00017] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.003 | 336 | tesseract 0.00024 | 0.0011 [0.00054, 0.003] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.007 | 336 | tesseract 0.018 | 0.022 [0.011, 0.046] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.003 | 720 | tesseract 0.00025 (bound) | 0.00034 [0.00013, 0.0011] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.007 | 720 | tesseract 0.012 | 0.02 [0.0073, 0.035] | 0.00 | tesseract / tesseract |  |
| stim_surface_unrotated_d3_p0.003 | 36 | tesseract 0.0059 | 0.019 [0.014, 0.028] | 0.00 | tesseract / mwpm |  |
| stim_surface_unrotated_d5_p0.003 | 200 | tesseract 0.0014 | 0.0053 [0.0033, 0.012] | 0.00 | tesseract / tesseract |  |
| stim_surface_unrotated_d7_p0.003 | 588 | tesseract 0.00016 | 0.001 [0.0005, 0.0035] | 0.00 | tesseract / tesseract |  |
