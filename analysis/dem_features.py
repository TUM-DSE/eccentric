"""Properties of each circuit's detector error model that need no decoding, for predicting its LER:
python -m analysis.dem_features <processes> <merged results.csv> <out.json>

For the circuits behind the rows of a collected sweep table: the probability that each detector fires
(exact for independent error mechanisms: (1 - prod(1 - 2p)) / 2 over the mechanisms that flip it), its
mean, median and maximum, the expected number of fired detectors per round, the total error probability,
and the number and mean detector count of the error mechanisms.
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import stim


def features(path):
    try:
        circuit = stim.Circuit.from_file(path)
        dem = circuit.detector_error_model(approximate_disjoint_errors=True)
        keep = np.ones(dem.num_detectors)
        mass, weights = 0.0, []
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
            keep[dets] *= 1 - 2 * p
            mass += p
            weights.append(len(dets))
        fire = (1 - keep) / 2
        return path, {"det_mean": float(fire.mean()), "det_median": float(np.median(fire)), "det_max": float(fire.max()),
                      "error_mass": mass, "mechanisms": len(weights), "mechanism_weight": float(np.mean(weights))}
    except Exception as e:
        return path, {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    processes, table, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    base = os.path.dirname(os.path.abspath(table))
    paths = {}
    for r in csv.DictReader(open(table)):
        if not r["result_file"]:
            continue
        run = os.path.join(base, "..", r["run"])
        with open(os.path.join(run, r["result_file"])) as f:
            needs = json.load(f)["job"]["needs"]
        paths[os.path.join(run, "circuits", needs + ".stim")] = f"{r['run']}/{needs}"
    with Pool(processes) as pool:
        result = {paths[p]: v for p, v in pool.imap_unordered(features, sorted(paths), chunksize=4)}
    with open(out, "w") as f:
        json.dump(result, f)
    print(len(result), "circuits,", sum("error" in v for v in result.values()), "failed")


if __name__ == "__main__":
    main()
