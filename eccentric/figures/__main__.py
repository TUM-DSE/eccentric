"""Build the figures of the ECCentric results: python -m eccentric.figures [--out site/figures] [--only id ...]

Writes, per figure, <id>.svg, <id>.png and <id>.csv (the plotted values), and manifest.json with each figure's
title, finding and caption and the dataset they were built from. The website copies this directory.
"""
import argparse
import datetime
import subprocess

from .. import dataset
from . import benchmark, featured, style


def _commit():
    """The last commit that changed the data the figures are built from"""
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h", "--", str(dataset.DEFAULT_PATH),
                               str(dataset.MATCHING_TIMING_PATH)], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eccentric.figures", description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="site/figures")
    parser.add_argument("--only", nargs="*", help="build only these figure functions (e.g. decoders willow)")
    args = parser.parse_args(argv)

    style.setup()
    table = dataset.load()
    rates = dataset.with_rates(table)
    source = {"dataset": "experiment_results/Sweep/results.csv", "schema_version": dataset.SCHEMA_VERSION,
              "results": int(len(rates)), "commit": _commit(), "built": datetime.date.today().isoformat()}
    reg = style.Registry(args.out, source)
    for build in featured.ALL + benchmark.ALL:
        if args.only and build.__name__ not in args.only:
            continue
        build(rates.copy(), reg)
        print("built", reg.entries[-1]["id"])
    reg.write_manifest()


if __name__ == "__main__":
    main()
