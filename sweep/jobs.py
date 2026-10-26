"""Create the jobs of sweep slices in a run directory: python -m sweep.jobs <run_dir> <slice> [<slice> ...]

Run them with the validation worker on as many machines as share the run directory:
    python -m validation.worker <run_dir> <parallel jobs>
"""
import json
import os
import sys

from sweep.slices import DECODE, SLICES

KEY = ("code", "schedule", "distance", "cycles", "backend", "backend_size", "error_type", "error_probability",
       "layout_method", "routing_method", "translating_method")


def config_id(slice_name, config):
    parts = [slice_name] + [f"{k}={config[k]}" for k in KEY if config.get(k) is not None]
    return ".".join(parts).replace("/", "_")


def make_jobs(slice_name):
    jobs = []
    for config in SLICES[slice_name]():
        decoders = config.pop("decoders")
        circuit_id = "circuit." + config_id(slice_name, config)
        jobs.append(dict(id=circuit_id, type="circuit", module="sweep.run_job", slice=slice_name, priority=0,
                         timeout=4 * 3600, **config))
        for decoder in decoders:
            jobs.append(dict(id=f"decode.{config_id(slice_name, config)}.decoder={decoder}", type="decode",
                             module="sweep.run_job", slice=slice_name, needs=circuit_id, decoder=decoder,
                             priority=2 if decoder == "tesseract" else 1, **config, **DECODE))
    return jobs


def main():
    run_dir, slices = sys.argv[1], sys.argv[2:]
    for sub in ("jobs", "claims", "results", "logs", "circuits"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    total = 0
    for slice_name in slices:
        jobs = make_jobs(slice_name)
        for job in jobs:
            with open(os.path.join(run_dir, "jobs", f"{job['id']}.json"), "w") as f:
                json.dump(job, f)
        circuits = sum(j["type"] == "circuit" for j in jobs)
        print(f"{slice_name}: {circuits} configurations, {len(jobs) - circuits} decodes")
        total += len(jobs)
    print(f"{total} jobs in {run_dir}")


if __name__ == "__main__":
    main()
