"""One decode of the live run, in ECCentric's environment: python decode_worker.py <spec.json> <progress.json>
ECCentric's adaptive decode (100 shots first, then doubling batches to the error target, shot limit or time
budget), with its progress written after every batch for run_live.py to watch"""
import json
import os
import sys
import time

import stim

from eccentric.decoders import decode
from validation.run_job import decode_adaptively

spec, out = json.load(open(sys.argv[1])), sys.argv[2]


def write(state):
    with open(out + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(out + ".tmp", out)


circuit = stim.Circuit.from_file(spec["circuit_file"])
write({"state": "started", "time": time.time()})
result = decode_adaptively(
    spec["settings"],
    lambda n: decode(spec["code"], circuit, n, spec["decoder"], spec["backend"], spec["error_type"]),
    lambda counts: write({"state": "running", **counts}))
write({"state": "failed" if result.get("failed") else "done", **result})
