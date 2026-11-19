"""A first training table from what ECCentric already saved, before Matchmaker's own extraction ran:
python scripts/bootstrap_from_legacy.py <ECCentric checkout> <out dir>

Uses ECCentric's experiment_results/Sweep/dem_features.json (six error-model features computed with the
same formulas as qec_matchmaker.features) and the size columns of its results.csv. Features ECCentric
counts differently (rounds) or did not compute are left NaN; scripts/update_from_eccentric.sh replaces
this table with one from qec_matchmaker's own extraction.
"""
import json
import math
import subprocess
import sys

import pandas as pd

from qec_matchmaker.dataset import circuit_key, from_eccentric, write_dataset
from qec_matchmaker.features import FEATURE_NAMES

RENAME = {"det_mean": "det_fire_mean", "det_median": "det_fire_median", "det_max": "det_fire_max",
          "error_mass": "error_mass", "mechanisms": "mechanisms", "mechanism_weight": "mechanism_weight_mean"}
COLUMNS = {"detectors": "detectors", "observables": "observables", "circuit_qubits": "qubits",
           "two_qubit_gates": "two_qubit_gates", "swaps": "swaps", "layers": "ticks", "measurements": "measurements",
           "undetectable_faults": "undetectable_faults", "undetectable_probability": "undetectable_probability"}


def main():
    eccentric, out = sys.argv[1], sys.argv[2]
    results_path = f"{eccentric}/experiment_results/Sweep/results.csv"
    legacy_path = f"{eccentric}/experiment_results/Sweep/dem_features.json"
    results = pd.read_csv(results_path, low_memory=False)
    legacy = json.load(open(legacy_path))
    store = {}
    for r in results[results.result_file.notna()].drop_duplicates(["run", "job_id"]).itertuples():
        key = circuit_key(r.run, r.job_id)
        if key in store or "error" in legacy.get(key, {"error": 1}):
            continue
        features = {name: math.nan for name in FEATURE_NAMES}
        features.update({new: legacy[key][old] for old, new in RENAME.items()})
        features.update({new: getattr(r, old) for old, new in COLUMNS.items()})
        features["error_mass_per_detector"] = features["error_mass"] / features["detectors"] if features["detectors"] else math.nan
        store[key] = features
    table, left_out = from_eccentric(results, store)
    commit = subprocess.run(["git", "-C", eccentric, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = write_dataset(table, out, {"eccentric_results": {"path": results_path, "commit": commit},
                                          "eccentric_dem_features": {"path": legacy_path, "commit": commit}}, left_out)
    print({k: manifest[k] for k in ("rows", "circuits", "codes", "left_out")})


if __name__ == "__main__":
    main()
