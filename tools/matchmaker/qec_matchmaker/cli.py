"""qec-matchmaker command line."""
import argparse
import json
import math
import os
import sys


# Inside the ECCentric repository: its sweep table and this tool's committed feature store
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ECCENTRIC_RESULTS = os.path.join(_ROOT, "..", "..", "experiment_results", "Sweep", "results.csv")
_STORE = os.path.join(_ROOT, "data", "features.jsonl.gz")
_GPU_PILOTS = os.path.join(_ROOT, "experiments", "gpu", "results")
_PACKAGED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def _read_results(path):
    """ECCentric's sweep table, read through eccentric.dataset when ECCentric is installed. Matchmaker needs only
    the columns from_eccentric uses, so a smaller table (e.g. one sweep's) is fine: no full-schema check"""
    try:
        from eccentric import dataset
    except ImportError:
        import pandas as pd
        return pd.read_csv(path, low_memory=False)
    return dataset.load(path, check=False)


def _eccentric_commit():
    import subprocess
    try:
        return subprocess.run(["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _features(args):
    from .api import features_of
    print(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in features_of(args.input).items()}, indent=1))


def _advise(args):
    from .api import advise
    speed = None
    if args.calibration:
        from .calibrate import load
        speed = load(args.calibration)
    advice = advise(args.input, objective=args.objective, budget_ms=args.budget_ms, per_round=args.per_round,
                    model_dir=args.models, speed_factors=speed, tolerance=args.tolerance, code_distance=args.distance,
                    gpu=args.gpu)
    print(json.dumps(advice.to_dict(), indent=1, default=str) if args.json else advice.summary())


def _extract(args):
    from .pipeline import extract
    keys = None
    if args.keys_from:
        import pandas as pd
        from .dataset import circuit_key
        results = pd.read_csv(args.keys_from, low_memory=False)
        results = results[results.result_file.notna()]
        keys = {circuit_key(r, j) for r, j in zip(results.run, results.job_id)}
    print(extract(args.run_dirs, args.store, processes=args.processes, keys=keys))


def _ingest(args):
    from .dataset import from_eccentric, write_dataset
    from .pipeline import features_of_store, load_store
    results = _read_results(args.results)
    timing = os.path.join(os.path.dirname(os.path.abspath(args.results)), "matching_timing.csv")
    if os.path.exists(timing) and not args.no_retiming:
        from .dataset import with_retiming
        results = with_retiming(results, timing)
    table, left_out = from_eccentric(results, features_of_store(load_store(args.store)))
    sources = {"eccentric_results": {"path": os.path.relpath(args.results, _ROOT), "commit": args.eccentric_commit or _eccentric_commit()},
               "feature_store": {"path": os.path.relpath(args.store, _ROOT)}}
    if "retimed_ms_per_shot" in results.columns:
        sources["retiming"] = {"path": os.path.relpath(timing, _ROOT), "rows": int(results.retimed_ms_per_shot.notna().sum())}
    manifest = write_dataset(table, args.out, sources, left_out)
    print(json.dumps({k: manifest[k] for k in ("rows", "circuits", "codes", "left_out")}))


def _train(args):
    from .dataset import load_dataset
    from .models import train
    table, manifest = load_dataset(args.data)
    bundle = train(table, manifest=manifest, calibrate=not args.no_calibration)
    if os.path.isdir(args.gpu_pilots):
        _add_gpu(bundle, args.gpu_pilots, args.store)
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics) as f:
            bundle.meta["metrics"] = json.load(f)
    bundle.save(args.out)
    print(f"trained on {manifest['rows']} rows ({manifest['circuits']} circuits), saved to {args.out}")


def _add_gpu(bundle, pilots_dir, store):
    import re
    from .gpu import add_gpu_model, load_pilots
    from .pipeline import features_of_store, load_store
    pilots = load_pilots(pilots_dir)
    codes = {k: m.group(1) for k in pilots.circuit_key.unique() if (m := re.search(r"\.code=([^.]+)", k))}
    report = add_gpu_model(bundle, pilots, features_of_store(load_store(store)), codes)
    best = report["variants"][report["chosen"]]
    print(f"GPU time model: {report['pilots']} pilots of {report['circuits']} circuits; on held-out codes "
          f"{best['within_3x']:.0%} within 3x, median error {10 ** best['median_abs_log10']:.2f}x")


def _train_gpu(args):
    from .models import ModelBundle
    bundle = ModelBundle.load(args.models)
    _add_gpu(bundle, args.pilots, args.store)
    bundle.save(args.models)


def _evaluate(args):
    from .dataset import load_dataset
    from .evaluate import evaluate, report_markdown
    table, manifest = load_dataset(args.data)
    options = {"parametric": False} if args.no_laws else {}
    metrics = evaluate(table, folds=args.folds, calibrate=not args.no_calibration, group_by=args.group_by, n_jobs=args.jobs,
                       **options)
    os.makedirs(args.out, exist_ok=True)
    suffix = ("" if args.group_by == "code" else f"_{args.group_by}") + ("_no_laws" if args.no_laws else "")
    with open(os.path.join(args.out, f"metrics{suffix}.json"), "w") as f:
        json.dump({"dataset": {k: manifest[k] for k in ("created", "rows", "circuits", "codes", "sources")}, **metrics},
                  f, indent=1, default=str)
    report = report_markdown(metrics)
    with open(os.path.join(args.out, f"REPORT{suffix}.md"), "w") as f:
        f.write(report)
    print(report)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qec-matchmaker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("features", help="print the features of a circuit, DEM or decoder-bench trace")
    p.add_argument("input")
    p.set_defaults(run=_features)

    p = sub.add_parser("advise", help="predict how decoding will go and recommend a decoder")
    p.add_argument("input", help=".stim circuit, .dem, or decoder-bench .h5 trace")
    p.add_argument("--objective", choices=["accurate", "cheapest", "budget"], default="cheapest")
    p.add_argument("--budget-ms", type=float, help="time per shot (or per round, with --per-round) for --objective budget")
    p.add_argument("--per-round", action="store_true")
    p.add_argument("--tolerance", type=float, default=1.25, help="LER slack of --objective cheapest")
    p.add_argument("--distance", type=float, help="the code's distance, if known: below threshold, the LER prediction then "
                                                   "follows the distance scaling beyond the training data")
    p.add_argument("--calibration", help="speed factors from qec_matchmaker.calibrate")
    p.add_argument("--gpu", action="store_true", help="a GPU is available: predict and recommend the GPU versions of "
                   "the decoders that have one (ECCentric's *_gpu decoders; times for one NVIDIA H200)")
    p.add_argument("--models", help="model directory (default: the packaged one)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=_advise)

    p = sub.add_parser("extract", help="add features of new circuit files of ECCentric run directories to a store")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--store", default=_STORE)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--keys-from", help="ECCentric results.csv: only the circuits its rows used")
    p.set_defaults(run=_extract)

    p = sub.add_parser("ingest", help="build the training table from ECCentric's results and the feature store")
    p.add_argument("--results", default=_ECCENTRIC_RESULTS, help="ECCentric's sweep table (default: this checkout's)")
    p.add_argument("--store", default=_STORE)
    p.add_argument("--out", default=os.path.join(_ROOT, "data"))
    p.add_argument("--eccentric-commit")
    p.add_argument("--no-retiming", action="store_true", help="ignore matching_timing.csv next to the results")
    p.set_defaults(run=_ingest)

    p = sub.add_parser("train", help="train the models on a training table")
    p.add_argument("--data", default=os.path.join(_ROOT, "data"))
    p.add_argument("--out", default=_PACKAGED)
    p.add_argument("--metrics", default="reports/metrics.json", help="evaluation metrics to store with the model")
    p.add_argument("--no-calibration", action="store_true", help="skip the conformal widening of the intervals")
    p.add_argument("--gpu-pilots", default=_GPU_PILOTS, help="GPU pilot results: also train the GPU time model")
    p.add_argument("--store", default=_STORE)
    p.set_defaults(run=_train)

    p = sub.add_parser("train-gpu", help="add the GPU time model, trained on GPU pilots, to a trained model")
    p.add_argument("--models", default=_PACKAGED, help="model directory to update (default: the packaged one)")
    p.add_argument("--pilots", default=_GPU_PILOTS)
    p.add_argument("--store", default=_STORE)
    p.set_defaults(run=_train_gpu)

    p = sub.add_parser("evaluate", help="cross-validate over held-out codes (or families, devices, noise models, or "
                                         "beyond the training range) and write a report")
    p.add_argument("--data", default=os.path.join(_ROOT, "data"))
    p.add_argument("--out", default=os.path.join(_ROOT, "reports"))
    p.add_argument("--folds", type=int, default=10, help="GroupKFold folds (more than groups: leave one group out)")
    p.add_argument("--group-by", default="code", choices=["code", "family", "backend", "error_type", "largest", "quietest",
                                                          "next_distance", "operations"],
                   help="what a fold holds out (largest, quietest, next_distance, operations: one split beyond the training range); "
                        "other than code, writes metrics_<group>.json and REPORT_<group>.md")
    p.add_argument("--no-laws", action="store_true", help="trees alone, without the scaling laws under them")
    p.add_argument("--jobs", type=int, default=1, help="folds trained in parallel")
    p.add_argument("--no-calibration", action="store_true")
    p.set_defaults(run=_evaluate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
