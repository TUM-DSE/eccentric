import argparse

from .runner import load_config, run_experiment


def main():
    parser = argparse.ArgumentParser(
        prog="python -m eccentric",
        description="Run ECCentric benchmark experiments described in YAML configuration files.",
    )
    parser.add_argument("configs", nargs="+", help="YAML configuration files")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of parallel worker processes (default: all CPUs)")
    parser.add_argument("-e", "--experiment", action="append", help="only run the experiment with this name (repeatable)")
    args = parser.parse_args()

    for path in args.configs:
        for experiment in load_config(path):
            if args.experiment and experiment["name"] not in args.experiment:
                continue
            run_experiment(experiment, args.workers)


if __name__ == "__main__":
    main()
