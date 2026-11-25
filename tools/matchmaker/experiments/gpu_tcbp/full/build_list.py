"""The circuits of Matchmaker's table (data/dataset.csv) not in the GPU slice, for tcbp_jobs_sc_gpu overnight
(nightly.sh): per circuit its key, code, backend, error type, the file its CPU decode jobs decoded (their
circuit_file, else <run>/circuits/<id>.stim) and those jobs' settings. python3 build_list.py <dataset.csv>
<slice.json> <out.json>"""
import csv, glob, json, os, sys
from collections import Counter
ROOT = "/home/manosgior/eccentric-validation"
table, slice_path, out_path = sys.argv[1:4]
done = {o["key"] for o in json.load(open(slice_path))}
first = {}
for r in csv.DictReader(open(table)):
    if r["circuit_key"] not in done and r["circuit_key"] not in first:
        first[r["circuit_key"]] = r
out, problems = [], Counter()
for key, r in sorted(first.items()):
    run, _, cid = key.partition("/")
    jobs = sorted(glob.glob(os.path.join(ROOT, run, "jobs", "decode" + cid[len("circuit"):] + ".decoder=*.json")))
    if not jobs:
        problems["no CPU decode job"] += 1; continue
    J = [json.load(open(j)) for j in jobs]
    files = {j.get("circuit_file") or os.path.join(ROOT, run, "circuits", j["needs"] + ".stim") for j in J}
    if len(files) != 1:
        problems["several circuit files"] += 1; continue
    f = files.pop()
    if not os.path.exists(f):
        problems["circuit file missing"] += 1; continue
    out.append({"key": key, "code": r["code"], "family": r["family"], "backend": r["backend"],
                "error_type": r["error_type"], "file": f})
json.dump(out, open(out_path, "w"), indent=0)
print(len(first), "circuits outside the slice;", len(out), "listed;", dict(problems))
