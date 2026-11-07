# Circuits from outside ECCentric

Matchmaker's packaged model (trained on ECCentric only) on Stim's generated memory circuits with uniform circuit noise and on decoder-bench's circuits with SI1000 noise, decoded and timed by ECCentric's sweep worker like the training data (target 100 errors, at most 200,000 shots or 10 minutes). The model is given each circuit's code distance. Errors are in decades (|log10 error|); intervals are the model's conformalized [q10, q90]. Trees alone, without the scaling laws: see EXTRAPOLATION.md.

| set | circuits | regime correct | best-LER median error | p90 | in [q10, q90] | time within 3x | choice good enough (accurate / cheapest) | always BP-OSD | always Tesseract |
|---|---|---|---|---|---|---|---|---|---|
| all | 34 | 100% | 0.08 | 0.28 | 86% | 84% | 100% / 94% | 67% | 100% |
| stim | 27 | 100% | 0.05 | 0.29 | 83% | 88% | 100% / 92% | 62% | 100% |
| decoder-bench | 7 | 100% | 0.17 | 0.21 | 100% | 68% | 100% / 100% | 80% | 100% |

Time per shot within 3x, per decoder (all circuits): mwpm 87% (23), bposd 82% (34), bplsd 82% (34), belief_find 85% (34), relay_bp 85% (34), tesseract 82% (34).

| circuit | detectors | best measured | predicted best LER [q10, q90] | P(above) | recommended (accurate / cheapest) | outside training |
|---|---|---|---|---|---|---|
| decoder_bench_bb_d12_p0.003 | 936 | tesseract 0.0017 | 0.013 [0.0031, 0.065] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_bb_d6_p0.003 | 252 | tesseract 0.074 | 0.12 [0.07, 0.27] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_bb_d6_p0.005 | 252 | tesseract 0.42 | 0.55 [0.26, 0.81] | 0.40 | tesseract / relay_bp |  |
| decoder_bench_color_d7_p0.003 | 252 | tesseract 0.0057 | 0.0084 [0.0044, 0.022] | 0.00 | tesseract / tesseract |  |
| decoder_bench_color_d7_p0.005 | 252 | tesseract 0.03 | 0.062 [0.04, 0.12] | 0.00 | tesseract / relay_bp |  |
| decoder_bench_surface_d7_p0.003 | 336 | tesseract 0.0038 | 0.0033 [0.002, 0.0096] | 0.00 | tesseract / tesseract |  |
| decoder_bench_surface_d7_p0.005 | 336 | tesseract 0.027 | 0.027 [0.011, 0.061] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.001 | 9 | tesseract 0.0031 | 0.0023 [0.0011, 0.0061] | 0.00 | tesseract / tesseract |  |
| stim_color_d3_p0.003 | 9 | tesseract 0.015 | 0.014 [0.0074, 0.03] | 0.00 | tesseract / relay_bp |  |
| stim_color_d5_p0.001 | 45 | tesseract 0.00052 | 0.00047 [0.00029, 0.0015] | 0.00 | tesseract / tesseract |  |
| stim_color_d5_p0.003 | 45 | tesseract 0.0096 | 0.0085 [0.0058, 0.021] | 0.00 | tesseract / relay_bp |  |
| stim_color_d7_p0.001 | 126 | tesseract 0.00011 | 7.4e-05 [4.7e-05, 0.0003] | 0.00 | tesseract / tesseract |  |
| stim_color_d7_p0.003 | 126 | tesseract 0.0039 | 0.0052 [0.0023, 0.011] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.001 | 270 | relay_bp 5.9e-05 | 1.6e-05 [1e-05, 3.2e-05] | 0.00 | tesseract / tesseract |  |
| stim_color_d9_p0.003 | 270 | tesseract 0.0014 | 0.0023 [0.0013, 0.0032] | 0.00 | tesseract / tesseract |  |
| stim_repetition_d11_p0.01 | 120 | mwpm 1.5e-05 | 0.0055 [0.0017, 0.011] | 0.00 | tesseract / mwpm |  |
| stim_repetition_d21_p0.01 | 440 | mwpm 1.9e-05 (bound) | 2.2e-05 [6.7e-06, 5.3e-05] | 0.00 | tesseract / mwpm |  |
| stim_repetition_d5_p0.01 | 24 | bposd 0.0015 | 0.016 [0.0054, 0.036] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d11_p0.003 | 1320 | mwpm 0.00021 | 5.8e-05 [3.6e-05, 0.00017] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d11_p0.007 | 1320 | tesseract 0.022 (bound) | 0.0095 [0.0039, 0.019] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d3_p0.001 | 24 | tesseract 0.00068 | 0.00067 [0.00042, 0.0017] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d3_p0.003 | 24 | tesseract 0.0061 | 0.006 [0.0041, 0.0086] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d3_p0.007 | 24 | tesseract 0.031 | 0.032 [0.018, 0.046] | 0.00 | tesseract / mwpm |  |
| stim_surface_rotated_d5_p0.001 | 120 | tesseract 7.5e-05 | 9.4e-05 [6.1e-05, 0.00031] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d5_p0.003 | 120 | tesseract 0.0019 | 0.0018 [0.001, 0.0052] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d5_p0.007 | 120 | tesseract 0.024 | 0.018 [0.011, 0.044] | 0.00 | tesseract / relay_bp |  |
| stim_surface_rotated_d7_p0.001 | 336 | tesseract 1.9e-05 (bound) | 1e-05 [5.7e-06, 5.9e-05] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.003 | 336 | tesseract 0.00024 | 0.00047 [0.00023, 0.0018] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d7_p0.007 | 336 | tesseract 0.018 | 0.013 [0.0063, 0.035] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.003 | 720 | tesseract 0.00025 (bound) | 0.0001 [5.9e-05, 0.00049] | 0.00 | tesseract / tesseract |  |
| stim_surface_rotated_d9_p0.007 | 720 | tesseract 0.012 | 0.0084 [0.0043, 0.022] | 0.00 | tesseract / tesseract |  |
| stim_surface_unrotated_d3_p0.003 | 36 | tesseract 0.0059 | 0.0099 [0.0064, 0.021] | 0.00 | tesseract / relay_bp |  |
| stim_surface_unrotated_d5_p0.003 | 200 | tesseract 0.0014 | 0.0029 [0.0011, 0.0075] | 0.00 | tesseract / tesseract |  |
| stim_surface_unrotated_d7_p0.003 | 588 | tesseract 0.00016 | 0.00045 [0.00026, 0.002] | 0.00 | tesseract / tesseract |  |
