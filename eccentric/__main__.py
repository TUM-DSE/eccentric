"""Command line: `eccentric <config.yaml> ...` runs experiments, `eccentric <command> ...` runs a command,
built in (`run`, `plugins`) or added by a plugin (eccentric.plugins, group eccentric.commands)."""
import argparse
import json
import os
import sys

from . import plugins


def run(argv):
    from .runner import load_config, run_experiment

    parser = argparse.ArgumentParser(
        prog="eccentric [run]",
        description="Run ECCentric benchmark experiments described in YAML configuration files.",
    )
    parser.add_argument("configs", nargs="+", help="YAML configuration files")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of parallel worker processes (default: all CPUs)")
    parser.add_argument("-e", "--experiment", action="append", help="only run the experiment with this name (repeatable)")
    args = parser.parse_args(argv)

    for path in args.configs:
        for experiment in load_config(path):
            if args.experiment and experiment["name"] not in args.experiment:
                continue
            run_experiment(experiment, args.workers)


def list_plugins(argv):
    parser = argparse.ArgumentParser(prog="eccentric plugins", description="List the installed ECCentric plugins.")
    parser.add_argument("--json", action="store_true", help="print as JSON")
    args = parser.parse_args(argv)
    found = plugins.describe()
    if args.json:
        print(json.dumps(found, indent=2))
        return
    for kind, entries in found.items():
        print(f"{kind}: " + ("none" if not entries else ""))
        for name, target in entries.items():
            print(f"  {name:24} {target}")


BUILTIN_COMMANDS = {"run": run, "plugins": list_plugins}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and not argv[0].startswith("-") and not os.path.exists(argv[0]):
        command = BUILTIN_COMMANDS.get(argv[0]) or plugins.get("commands", argv[0])
        if command is not None:
            return command(argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        commands = sorted(set(BUILTIN_COMMANDS) | set(plugins.names("commands")))
        print("usage: eccentric <config.yaml> ... | eccentric <command> ...\n\ncommands: " + ", ".join(commands)
              + "\n\n`eccentric run -h` shows the options for running experiments.")
        return
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
