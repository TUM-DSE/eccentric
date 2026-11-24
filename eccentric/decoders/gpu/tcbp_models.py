"""TCBP models of tcbp_jobs_gpu (nvqldpc.tcbp_jobs_decoder), one per code family.

A TCBP model (Howe, Maurya et al., arXiv:2611.21321, preprint) is fitted from a check matrix alone, with training
error rates drawn per fault mechanism (init_tcbp_model's defaults: log-uniform in [1e-4, 1e-2], seed 1), and is
reused for the other distances, error rates and devices of its family, as in the paper. A family is a code name
(the "code" of a sweep job). Its model is fitted on the family's reference circuit: the circuit with the most
fault mechanisms among the given ones (ties: the first path in sorted order), so the choice does not depend on the
order in which jobs run. A model fitted on the family's smallest circuit decoded its larger ones worse than their
own models (23 Nov 2026, Matchmaker slice: surface d13 7 vs 0 failures, d5 1,366 vs 1,267 in 20,000 shots). The models live in MODEL_DIR (or $ECCENTRIC_TCBP_MODELS) as <code>.tcbp, with
<code>.json recording the reference circuit and the fit report.

    python -m eccentric.decoders.gpu.tcbp_models fit <code>=<circuit.stim> [<code>=<circuit.stim> ...] [--force]
    python -m eccentric.decoders.gpu.tcbp_models fit-slice <slice.json> [--force]   (Matchmaker's slice format)
    python -m eccentric.decoders.gpu.tcbp_models fit-table <dataset.csv> <runs root> [--force]
        (Matchmaker's table, tools/matchmaker/data/dataset.csv: the reference of each code is its circuit with the
        most mechanisms in the table; circuit_key <run>/<circuit> is the file <runs root>/<run>/circuits/<circuit>.stim)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import stim

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcbp_models")


def model_dir() -> str:
    return os.environ.get("ECCENTRIC_TCBP_MODELS", MODEL_DIR)


def model_path(code: str) -> str:
    return os.path.join(model_dir(), f"{code}.tcbp")


def load(code: str) -> bytes:
    path = model_path(code)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no TCBP model for code family {code!r} at {path}; fit one with "
                                f"python -m eccentric.decoders.gpu.tcbp_models fit {code}=<circuit.stim>")
    with open(path, "rb") as f:
        return f.read()


def check_matrix(circuit: stim.Circuit, approximate_disjoint_errors: bool = False):
    from ..bp_osd import dem_to_check_matrices
    dem = circuit.detector_error_model(approximate_disjoint_errors=approximate_disjoint_errors)
    chk, _, _ = dem_to_check_matrices(dem, xor_priors=True)
    return chk.tocsr()


def fit(circuit: stim.Circuit):
    """The TCBP model (bytes) and fit report of a circuit's check matrix, with init_tcbp_model's defaults"""
    from cudaq_qec.plugins import nv_qldpc
    from . import runtime
    runtime.cupy()
    return nv_qldpc.init_tcbp_model(check_matrix(circuit, approximate_disjoint_errors=True), return_report=True)


def fit_reference(code: str, reference: str, force: bool = False, **note) -> str:
    """Fit the model of family `code` on its reference circuit file and write it to model_dir()"""
    out = model_path(code)
    if os.path.exists(out) and not force:
        return f"{code}: kept {out}"
    circuit = stim.Circuit.from_file(reference)
    start = time.time()
    model, report = fit(circuit)
    seconds = time.time() - start
    os.makedirs(model_dir(), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        f.write(model)
    os.replace(out + ".tmp", out)
    with open(os.path.join(model_dir(), f"{code}.json"), "w") as f:
        json.dump({"code": code, "reference_circuit": os.path.basename(reference),
                   "reference_sha256": hashlib.sha256(open(reference, "rb").read()).hexdigest(),
                   "detectors": circuit.num_detectors, "fit_seconds": round(seconds, 1), **note, "report": report},
                  f, indent=1, default=str)
    return f"{code}: fitted on {os.path.basename(reference)} in {seconds:.1f} s"


def fit_table(table: str, root: str, force: bool = False, codes=None):
    """Fit every code of Matchmaker's dataset.csv (or those in `codes`) on its circuit with the most mechanisms"""
    import csv
    best = {}
    with open(table) as f:
        for row in csv.DictReader(f):
            if not row.get("mechanisms"):
                continue
            key, m = row["circuit_key"], int(float(row["mechanisms"]))
            if row["code"] not in best or (-m, key) < (-best[row["code"]][0], best[row["code"]][1]):
                best[row["code"]] = (m, key)
    for code in sorted(best):
        if codes and code not in codes:
            continue
        m, key = best[code]
        try:
            yield fit_reference(code, circuit_file(root, key), force, fault_mechanisms=m, circuit_key=key,
                                chosen_from=os.path.basename(table))
        except Exception as e:  # reported, the other codes go on
            yield f"{code}: FAILED {e!r}"


def circuit_file(root: str, key: str) -> str:
    """The file of circuit_key <run>/<circuit>: in <run>'s circuits, or, for runs that decode other runs' circuits
    (sweep_bposd_tuned), the circuit_file its decode jobs name. Not a search by name: runs hold different circuits
    under the same name (sweep_idle's have idle noise, sweep1's and sweep_bicycle's do not)"""
    import glob
    run, _, name = key.partition("/")
    path = os.path.join(root, run, "circuits", name + ".stim")
    if os.path.exists(path):
        return path
    jobs = sorted(glob.glob(os.path.join(root, run, "jobs", "decode" + name[len("circuit"):] + ".decoder=*.json")))
    files = {json.load(open(j)).get("circuit_file") for j in jobs} - {None}
    if len(files) != 1:
        raise FileNotFoundError(f"{key}: {len(jobs)} decode jobs name {len(files)} circuit files")
    return files.pop()


def fit_family(code: str, files, force: bool = False) -> str:
    """Fit the model of family `code` on the reference circuit among `files` and write it to model_dir()"""
    out = model_path(code)
    if os.path.exists(out) and not force:
        return f"{code}: kept {out}"
    sizes = {}
    for file in sorted(set(files)):
        circuit = stim.Circuit.from_file(file)
        # approximate_disjoint_errors: device noise models use PAULI_CHANNEL_1 (only the matrix's structure matters)
        sizes[file] = (circuit.detector_error_model(approximate_disjoint_errors=True).num_errors, circuit)
    reference = min(sizes, key=lambda f: (-sizes[f][0], f))
    mechanisms, circuit = sizes[reference]
    start = time.time()
    model, report = fit(circuit)
    seconds = time.time() - start
    os.makedirs(model_dir(), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        f.write(model)
    os.replace(out + ".tmp", out)
    with open(os.path.join(model_dir(), f"{code}.json"), "w") as f:
        json.dump({"code": code, "reference_circuit": os.path.basename(reference),
                   "reference_sha256": hashlib.sha256(open(reference, "rb").read()).hexdigest(),
                   "fault_mechanisms": mechanisms, "detectors": circuit.num_detectors,
                   "candidates": len(sizes), "fit_seconds": round(seconds, 1), "report": report}, f, indent=1,
                  default=str)
    return f"{code}: fitted on {os.path.basename(reference)} ({mechanisms} mechanisms) in {seconds:.1f} s"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    one = sub.add_parser("fit", help="fit families from <code>=<circuit.stim> pairs (several per code allowed)")
    one.add_argument("pairs", nargs="+")
    one.add_argument("--force", action="store_true", help="refit families that already have a model")
    many = sub.add_parser("fit-slice", help="fit every family of a Matchmaker slice.json (entries with code, file)")
    many.add_argument("slice")
    many.add_argument("--force", action="store_true")
    table = sub.add_parser("fit-table", help="fit every code of Matchmaker's dataset.csv")
    table.add_argument("table")
    table.add_argument("root", help="directory holding the sweep run directories")
    table.add_argument("--force", action="store_true")
    table.add_argument("--codes", nargs="*", help="only these codes")
    args = ap.parse_args(argv)
    if args.command == "fit-table":
        for line in fit_table(args.table, args.root, args.force, args.codes):
            print(line, flush=True)
        return
    families = defaultdict(list)
    if args.command == "fit":
        for pair in args.pairs:
            code, _, file = pair.partition("=")
            families[code].append(file)
    else:
        for entry in json.load(open(args.slice)):
            families[entry["code"]].append(entry["file"])
    for code in sorted(families):
        print(fit_family(code, families[code], args.force), flush=True)


if __name__ == "__main__":
    sys.exit(main())
