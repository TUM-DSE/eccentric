"""Summarize a validation run: python -m validation.report <run_dir> <out_dir>"""
import json
import math
import os
import sys

import pandas as pd

SIGMAS = 3


def load(run_dir):
    rows = []
    for name in os.listdir(os.path.join(run_dir, "results")):
        if name.endswith(".json"):
            with open(os.path.join(run_dir, "results", name)) as f:
                r = json.load(f)
            job = r.pop("job")
            rows.append({**{f"job_{k}": v for k, v in job.items()}, **r})
    return pd.DataFrame(rows)


def z_score(e1, n1, e2, n2):
    if not n1 or not n2:
        return float("nan")
    p = (e1 + e2) / (n1 + n2)
    if p in (0, 1):
        return 0.0
    return (e1 / n1 - e2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def phase1(df):
    from eccentric.codes.catalog import CODES, FixedCode

    circuits = df[df.job_type == "circuit"]
    rows = []
    for _, r in circuits.iterrows():
        code, schedule = r.job_code, r.job_schedule
        spec = CODES[code]
        d = spec.distance if isinstance(spec, FixedCode) else spec.min_d + spec.step
        tag = f"{code}.{schedule}" if isinstance(schedule, str) else code
        dist = df[df.job_id == f"circuit_distance.{tag}"]
        exact = df[df.job_id == f"code_distance_exact.{code}"]
        est = df[df.job_id == f"code_distance_estimate.{code}"]
        row = {
            "code": code, "schedule": schedule if isinstance(schedule, str) else "",
            "d": d, "qubits_ok": r.num_qubits == r.expected_num_qubits, "deterministic": r.deterministic,
            "observables": r.observables, "expected_observables": r.expected_observables,
            "layers": r.layers,
        }
        row["code_distance"] = next((x.distance for x in (exact, est) if len(x) and pd.notna(x.iloc[0].get("distance"))
                                     for x in [x.iloc[0]]), None)
        row["code_distance_method"] = next((x.iloc[0].method for x in (exact, est) if len(x) and pd.notna(x.iloc[0].get("distance"))), "")
        if len(dist):
            x = dist.iloc[0]
            row["circuit_distance"] = x.get("circuit_distance_upper_bound") if x.status == "done" else None
            row["circuit_distance_note"] = "" if x.status == "done" else x.error
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["code", "schedule"])
    problems = out[~out.qubits_ok | ~out.deterministic
                   | (out.expected_observables.notna() & (out.observables != out.expected_observables))
                   | (out.code_distance.notna() & (out.code_distance.astype(float) < out.d))]
    return out, problems


def phase3(df):
    dec = df[df.job_type.isin(["decode", "decode_v1"])].copy()
    dec["ok"] = dec.status == "done"
    rows = []
    for (code, p), g in dec.groupby(["job_code", "job_p"]):
        res = {r.job_decoder: r for _, r in g.iterrows()}
        ref = res.get("bposd_batch")
        tess = res.get("tesseract")
        for name, r in res.items():
            row = {"code": code, "p": p, "decoder": name, "status": r.status,
                   "shots": r.get("shots"), "errors": r.get("errors"), "ler": r.get("logical_error_rate"),
                   "seconds": r.get("seconds")}
            if r.status == "done" and ref is not None and ref.status == "done" and name != "bposd_batch":
                row["z_vs_bposd"] = z_score(r.errors, r.shots, ref.errors, ref.shots)
            if r.status == "done" and tess is not None and tess.status == "done" and name != "tesseract":
                row["z_vs_tesseract"] = z_score(r.errors, r.shots, tess.errors, tess.shots)
            twin = res.get(name.removesuffix("_v1")) if name.endswith("_v1") else None
            if twin is not None and r.status == "done" and twin.status == "done":
                row["z_v1_vs_v2"] = z_score(r.errors, r.shots, twin.errors, twin.shots)
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["code", "p", "decoder"])
    flags = out[(out.get("z_vs_tesseract", pd.Series(dtype=float)) < -SIGMAS)
                | (out.get("z_v1_vs_v2", pd.Series(dtype=float)).abs() > SIGMAS)]
    return out, flags


def main():
    run_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    df = load(run_dir)
    p1, p1_problems = phase1(df)
    p3, p3_flags = phase3(df)
    p1.to_csv(os.path.join(out_dir, "phase1_circuits.csv"), index=False)
    p3.to_csv(os.path.join(out_dir, "phase3_decoders.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"Phase 1: {len(p1)} circuits, {len(p1_problems)} with problems")
    print(p1_problems.to_string(index=False) if len(p1_problems) else "  none")
    print(f"\nPhase 3: {len(p3)} decodes, {len(p3_flags)} flagged (beats Tesseract or v1 != v2 by >{SIGMAS} sigma)")
    print(p3_flags.to_string(index=False) if len(p3_flags) else "  none")


if __name__ == "__main__":
    main()
