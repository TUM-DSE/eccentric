import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import yaml

from .codes import get_min_n
from .experiments import EXPERIMENTS, Skip, Task
from .utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging

SWEEPS = {
    "backends": "backend",
    "codes": "code",
    "decoders": "decoder",
    "error_types": "error_type",
    "error_probabilities": "error_probability",
    "layout_methods": "layout_method",
    "routing_methods": "routing_method",
    "translating_methods": "translating_method",
    "gate_sets": "gate_set",
    "optimization_levels": "optimization_level",
}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)["experiments"]


def experiment_type(experiment):
    name = experiment.get("type", "logical_error")
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment type '{name}', choose from: {', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def expand(experiment):
    module = experiment_type(experiment)
    missing = [key for key in ("name", "num_samples", *module.REQUIRED) if key not in experiment]
    if missing:
        raise ValueError(f"Experiment '{experiment.get('name')}' is missing: {', '.join(missing)}")
    if "backends_sizes" in experiment and "distances" in experiment:
        raise ValueError(f"Experiment '{experiment['name']}' cannot set both backends_sizes and distances")

    defaults = getattr(module, "DEFAULTS", {})
    axes = {field: experiment.get(key, defaults.get(key, [None])) for key, field in SWEEPS.items()}
    sizes = experiment.get("backends_sizes", [None])
    distances = experiment.get("distances", [None])

    tasks = []
    for *values, size, d in product(*axes.values(), sizes, distances):
        params = dict(zip(axes, values))
        if d is not None:
            size = get_min_n(params["code"], d)
        tasks.append(Task(
            experiment=experiment["name"],
            num_samples=experiment["num_samples"],
            cycles=experiment.get("cycles"),
            backend_size=size,
            distance=d,
            **params,
        ))
    return module, tasks


def run_experiment(experiment, workers=None):
    module, tasks = expand(experiment)
    name = experiment["name"]
    setup_experiment_logging(name)
    save_experiment_metadata(experiment, name)
    print(f"{name}: running {len(tasks)} configurations")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(module.run, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                rows = future.result()
            except Skip as e:
                logging.info(f"{name} | Skipped {task.describe()}: {e}")
                status = "skipped"
            except Exception as e:
                logging.error(f"{name} | Failed {task.describe()}: {e!r}")
                status = "failed"
            else:
                for row in rows:
                    save_results_to_csv(row, name)
                    logging.info(f"{name} | {row}")
                status = "done"
            print(f"[{done}/{len(tasks)}] {status}: {task.describe()}")
