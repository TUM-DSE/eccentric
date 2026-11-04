"""qec-matchmaker command line."""
import argparse
import json
import math
import os
import sys


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
                    model_dir=args.models, speed_factors=speed, tolerance=args.tolerance)
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
    import pandas as pd
    from .dataset import from_eccentric, write_dataset
    from .pipeline import features_of_store, load_store
    results = pd.read_csv(args.results, low_memory=False)
    table, left_out = from_eccentric(results, features_of_store(load_store(args.store)))
    sources = {"eccentric_results": {"path": args.results, "commit": args.eccentric_commit}, "feature_store": {"path": args.store}}
    manifest = write_dataset(table, args.out, sources, left_out)
    print(json.dumps({k: manifest[k] for k in ("rows", "circuits", "codes", "left_out")}))


def _train(args):
    from .dataset import load_dataset
    from .models import train
    table, manifest = load_dataset(args.data)
    bundle = train(table, manifest=manifest, calibrate=not args.no_calibration)
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics) as f:
            bundle.meta["metrics"] = json.load(f)
    bundle.save(args.out)
    print(f"trained on {manifest['rows']} rows ({manifest['circuits']} circuits), saved to {args.out}")


def _evaluate(args):
    from .dataset import load_dataset
    from .evaluate import evaluate, report_markdown
    table, manifest = load_dataset(args.data)
    metrics = evaluate(table, folds=args.folds, calibrate=not args.no_calibration)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"dataset": {k: manifest[k] for k in ("created", "rows", "circuits", "codes", "sources")}, **metrics},
                  f, indent=1, default=str)
    report = report_markdown(metrics)
    with open(os.path.join(args.out, "REPORT.md"), "w") as f:
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
    p.add_argument("--calibration", help="speed factors from qec_matchmaker.calibrate")
    p.add_argument("--models", help="model directory (default: the packaged one)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=_advise)

    p = sub.add_parser("extract", help="add features of new circuit files of ECCentric run directories to a store")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--store", required=True)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--keys-from", help="ECCentric results.csv: only the circuits its rows used")
    p.set_defaults(run=_extract)

    p = sub.add_parser("ingest", help="build the training table from ECCentric's results and the feature store")
    p.add_argument("--results", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--out", default="data")
    p.add_argument("--eccentric-commit")
    p.set_defaults(run=_ingest)

    p = sub.add_parser("train", help="train the models on a training table")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
    p.add_argument("--metrics", default="reports/metrics.json", help="evaluation metrics to store with the model")
    p.add_argument("--no-calibration", action="store_true", help="skip the conformal widening of the intervals")
    p.set_defaults(run=_train)

    p = sub.add_parser("evaluate", help="cross-validate over held-out codes and write a report")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--no-calibration", action="store_true")
    p.set_defaults(run=_evaluate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    sys.exit(main())
