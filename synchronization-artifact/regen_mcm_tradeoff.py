"""
regen_mcm_tradeoff.py
=====================
Regenerate the MCM readout-fidelity tradeoff CSVs that feed the four-panel
figure data/mcm_qec_evaluation.pdf (utils/mcm_plot.ipynb), comparing the CNN
and HERQULES mid-circuit-measurement decoders.

For each system the readout measure_error per readout length is
    measure_error = 1 - geomean(Q1, Q3, Q4, Q5)    (Q2 excluded)
from the per-qubit accuracy tables below. HERQULES = the NO-RETRAIN / truncated
table (the corrected competitor baseline).

ibm vs futuristic differ ONLY in coherence (T1/T2 x3); same gate error, same
readout measure_errors. Distances 7/9/11, surface-code lattice-surgery logical
measurement (sim/circuit_4), 100k shots.

Writes (into the artifact dir; copy to experiment_results/ afterwards):
  mcm_tradeoff_mcm_ibm.csv          mcm_tradeoff_mcm_futuristic_2.csv
  mcm_tradeoff_herqules_ibm.csv     mcm_tradeoff_herqules_futuristic_2.csv

Env:
  VALIDATE=1  -> reproduce the OLD herqules_ibm pairs and diff vs the existing
                 CSV (sanity-check the noise params); writes nothing permanent.
  PROCS=N     -> cap worker processes (default cpu_count).
  DURATIONS   -> comma list override (default 200,250,400,500,600,750,800,1000).
"""

import sys, os, csv
import multiprocessing as mp

# Per-qubit readout accuracies (Q1, Q3, Q4, Q5) -- Q2 excluded.
# measure_error = 1 - geomean(Q1, Q3, Q4, Q5).
CNN_ACC = {
    200:  (0.782, 0.864, 0.774, 0.770),
    250:  (0.835, 0.895, 0.833, 0.856),
    400:  (0.907, 0.926, 0.901, 0.951),
    500:  (0.939, 0.934, 0.927, 0.968),
    600:  (0.953, 0.938, 0.937, 0.970),
    750:  (0.964, 0.940, 0.943, 0.970),
    800:  (0.965, 0.941, 0.944, 0.970),
    1000: (0.970, 0.942, 0.947, 0.970),
}
# HERQULES no-retrain / truncated (the corrected competitor baseline).
HERQULES_ACC = {
    200:  (0.500, 0.500, 0.500, 0.500),
    250:  (0.500, 0.506, 0.500, 0.500),
    400:  (0.500, 0.847, 0.512, 0.722),
    500:  (0.513, 0.915, 0.631, 0.936),
    600:  (0.590, 0.920, 0.768, 0.965),
    750:  (0.831, 0.922, 0.895, 0.969),
    800:  (0.887, 0.923, 0.921, 0.970),
    1000: (0.969, 0.925, 0.942, 0.970),
}
SYSTEMS = {"mcm": CNN_ACC, "herqules": HERQULES_ACC}


def measure_error(acc, ns):
    q1, q3, q4, q5 = acc[ns]
    return 1.0 - (q1 * q3 * q4 * q5) ** 0.25


DISTANCES = [7, 9, 11]
CNOT_LATENCY_NS = 70
CNOT_ERROR = 0.0002
IDLE_MUL = 3
BASIS, LS_BASIS = "Z", "X"
NUM_SHOTS = 100_000

# Coherence presets (us). ibm = base; futuristic = base x3.
PRESETS = {
    "ibm":        dict(t1=130.0,     t2=170.0,     suffix="ibm"),
    "futuristic": dict(t1=130.0 * 3, t2=170.0 * 3, suffix="futuristic_2"),
}

_DUR_ENV = os.environ.get("DURATIONS")
DURATIONS = ([int(x) for x in _DUR_ENV.split(",")] if _DUR_ENV
             else [200, 250, 400, 500, 600, 750, 800, 1000])
PROCS = int(os.environ.get("PROCS", "0")) or mp.cpu_count()


def _run_one(args):
    d, latency_ns, merr, t1, t2 = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sim"))
    from circuit_4 import circuit  # noqa: PLC0415
    sim = circuit(
        distance=d, num_patches_x=20, num_patches_y=20, spacing=1,
        disable_noise=False,
        fixed_measure_latency=latency_ns, fixed_measure_noise=merr,
        fixed_cnot_latency=CNOT_LATENCY_NS, fixed_cnot_noise=CNOT_ERROR,
        fixed_t1=t1, fixed_t2=t2, idle_multiplier=IDLE_MUL,
        merge=True, basis=BASIS, ls_basis=LS_BASIS, sync=None,
        rounds_per_op=d + 1,
    ).from_string("qreg q[2];")
    ler, _ = sim.get_error_rate(ckt=sim.ckt, num_shots=NUM_SHOTS)
    return (d, latency_ns, merr, ler)


def _gen(pairs, t1, t2):
    """pairs: list of (latency_ns, measure_error). Returns sorted rows."""
    configs = [(d, lat, err, t1, t2) for d in DISTANCES for lat, err in pairs]
    with mp.Pool(PROCS) as pool:
        rows = list(pool.imap_unordered(_run_one, configs))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "measure_latency_ns", "measure_error", "logical_error_rate"])
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def validate():
    """Reproduce the OLD herqules_ibm pairs and diff vs the existing CSV."""
    old_pairs = [(250, 0.5), (500, 0.292), (750, 0.104), (1000, 0.081)]
    p = PRESETS["ibm"]
    rows = _gen(old_pairs, p["t1"], p["t2"])
    ref = {}
    refpath = "mcm_tradeoff_herqules_ibm.csv"
    for r in csv.DictReader(open(refpath)):
        ref[(int(r["distance"]), int(r["measure_latency_ns"]))] = float(r["logical_error_rate"])
    print(f"{'d':>3} {'lat':>5} {'err':>7} {'new_LER':>9} {'old_LER':>9} {'dabs':>8}")
    maxd = 0.0
    for d, lat, err, ler in rows:
        o = ref.get((d, lat), float("nan"))
        dd = abs(ler - o)
        maxd = max(maxd, dd)
        print(f"{d:>3} {lat:>5} {err:>7.3f} {ler:>9.5f} {o:>9.5f} {dd:>8.5f}")
    print(f"\nMAX |new-old| = {maxd:.5f}  (Monte-Carlo noise at 100k shots ~ few e-3)")


def main():
    if os.environ.get("VALIDATE"):
        validate()
        return
    print(f"durations={DURATIONS} procs={PROCS} shots={NUM_SHOTS}", flush=True)
    for sysname, acc in SYSTEMS.items():
        pairs = [(ns, measure_error(acc, ns)) for ns in DURATIONS]
        for pname, p in PRESETS.items():
            rows = _gen(pairs, p["t1"], p["t2"])
            _write(f"mcm_tradeoff_{sysname}_{p['suffix']}.csv", rows)


if __name__ == "__main__":
    main()
