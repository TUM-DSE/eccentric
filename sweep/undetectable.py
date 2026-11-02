"""Count the undetectable logical faults of the circuits that runs built before their circuit jobs did:
python -m sweep.undetectable <processes> <run_dir> [<run_dir> ...]

Writes <run_dir>/undetectable.json, keyed by circuit job id, which sweep.collect reads for circuits
whose results lack the counts.
"""
import json
import os
import sys
from multiprocessing import Pool

import stim

from sweep.run_job import undetectable_faults


def count(path):
    try:
        return os.path.basename(path)[:-len(".stim")], undetectable_faults(stim.Circuit.from_file(path))
    except Exception as e:
        return os.path.basename(path)[:-len(".stim")], {"undetectable_error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    processes, run_dirs = int(sys.argv[1]), sys.argv[2:]
    with Pool(processes) as pool:
        for run_dir in run_dirs:
            folder = os.path.join(run_dir, "circuits")
            paths = [os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".stim")] if os.path.isdir(folder) else []
            counts = dict(pool.imap_unordered(count, paths, chunksize=4))
            with open(os.path.join(run_dir, "undetectable.json"), "w") as f:
                json.dump(counts, f)
            hit = sum(1 for c in counts.values() if c.get("undetectable_faults"))
            print(f"{run_dir}: {len(counts)} circuits, {hit} with undetectable logical faults", flush=True)


if __name__ == "__main__":
    main()
