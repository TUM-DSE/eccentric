"""Verification plan: CPU shots per (circuit, decoder) and chunk commands; GPU runs on all 10^6 shots"""
import sys
M = 1_000_000
# single-core CPU shots/s measured on steve (cpu_rate.py, 200 shots)
RATE = {
 ("surface_d5_p002", "bposd"): 36, ("bb72_d6_p002", "bposd"): 1537, ("gross_d12_p001", "bposd"): 440, ("hgp_hamming_d3_p001", "bposd"): 126,
 ("surface_d5_p002", "bposd_tuned"): 373, ("color_d5_p002", "bposd_tuned"): 240, ("bb72_d6_p002", "bposd_tuned"): 290,
 ("hgp_hamming_d3_p001", "bposd_tuned"): 2129, ("mhc36_d4_p001", "bposd_tuned"): 1267,
 ("surface_d5_p002", "relay_bp"): 494, ("color_d5_p002", "relay_bp"): 294, ("bb72_d6_p002", "relay_bp"): 476, ("gross_d12_p001", "relay_bp"): 122,
 ("hgp_hamming_d3_p001", "relay_bp"): 2403, ("mhc36_d4_p001", "relay_bp"): 619, ("steane_d9_p001", "relay_bp"): 15.5,
 ("surface_d5_p002", "union_find"): 3566, ("color_d5_p002", "union_find"): 3046, ("bb72_d6_p002", "union_find"): 490, ("gross_d12_p001", "union_find"): 65,
 ("hgp_hamming_d3_p001", "union_find"): 9474, ("steane_d9_p001", "union_find"): 447, ("mhc36_d4_p001", "union_find"): 7474,
 ("surface_d5_p002", "gari"): 6654, ("color_d5_p002", "gari"): 7388, ("bb72_d6_p002", "gari"): 3454, ("gross_d12_p001", "gari"): 577,
 ("hgp_hamming_d3_p001", "gari"): 2206, ("steane_d9_p001", "gari"): 117, ("mhc36_d4_p001", "gari"): 840,
 ("surface_d5_p002", "bplsd"): 36, ("color_d5_p002", "bplsd"): 13, ("bb72_d6_p002", "bplsd"): 1533, ("gross_d12_p001", "bplsd"): 448, ("hgp_hamming_d3_p001", "bplsd"): 129,
 ("surface_d5_p002", "belief_find"): 37, ("color_d5_p002", "belief_find"): 13, ("bb72_d6_p002", "belief_find"): 1554, ("gross_d12_p001", "belief_find"): 445, ("hgp_hamming_d3_p001", "belief_find"): 128,
 ("mhc36_d4_p001", "hier_bposd"): 403, ("steane_d9_p001", "hier_bposd"): 10, ("mhc36_d4_p001", "hier_bposd_hard"): 269, ("steane_d9_p001", "hier_bposd_hard"): 6.6,
}
SHOTS = {k: M for k in RATE}
SHOTS.update({("surface_d5_p002", "bposd"): 100_000, ("gross_d12_p001", "union_find"): 200_000, ("steane_d9_p001", "relay_bp"): 50_000,
              ("surface_d5_p002", "bplsd"): 100_000, ("surface_d5_p002", "belief_find"): 100_000,
              ("color_d5_p002", "bplsd"): 50_000, ("color_d5_p002", "belief_find"): 50_000,
              ("steane_d9_p001", "hier_bposd"): 50_000, ("steane_d9_p001", "hier_bposd_hard"): 50_000})
if __name__ == "__main__":
    what = sys.argv[1]
    if what == "cpu":
        jobs = []
        for (tag, dec), n in SHOTS.items():
            chunk = max(1000, int(RATE[(tag, dec)] * 600) // 1000 * 1000)
            for a in range(0, n, chunk):
                jobs.append((n / RATE[(tag, dec)], f"{tag} {dec} {a} {min(n, a + chunk)}"))
        for _, j in sorted(jobs, key=lambda x: -x[0]):
            print(j)
    else:
        for (tag, dec) in sorted(SHOTS, key=lambda k: (k[1], k[0])):
            print(tag, dec)
