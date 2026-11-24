"""slice.json with each entry's file = the circuit its CPU decode jobs decoded (job circuit_file, else <run>/circuits)"""
import json, glob, os, sys
ROOT = "/home/manosgior/eccentric-validation"
src, dst = sys.argv[1], sys.argv[2]
s = json.load(open(src)); changed = []
for i, o in enumerate(s):
    run, _, cid = o["key"].partition("/")
    jobs = sorted(glob.glob(os.path.join(ROOT, run, "jobs", "decode" + cid[len("circuit"):] + ".decoder=*.json")))
    files = {json.load(open(j)).get("circuit_file") or os.path.join(ROOT, run, "circuits", cid + ".stim") for j in jobs}
    assert len(files) == 1, (i, files)
    f = files.pop(); assert os.path.exists(f), f
    if f != o["file"]:
        changed.append(i); o["file_before_fix"] = o["file"]; o["file"] = f
json.dump(s, open(dst, "w"), indent=1)
print("changed", len(changed), changed)
