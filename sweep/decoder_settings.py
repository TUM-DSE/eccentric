"""Pilot of decoder settings: the BP-type decoders with ECCentric's settings and with alternatives, on the
same shots of a sample of sweep circuits, so that every comparison is paired.

    python -m sweep.decoder_settings make <run_dir> <results.csv> <runs base dir> [<circuits>]
    python -m sweep.decoder_settings report <run_dir> <out.csv>

Jobs run through sweep.run_job (types settings_shots and settings_decode) with the usual workers. A shots
job samples a circuit's shots once (fixed seed) into shots/; a decode job decodes them in order with one
decoder setting until they are done or its CPU budget is spent, and keeps every shot's outcome in
outcomes/ for the paired comparison. Check matrices and priors are those of ECCentric's decoders
(dem_to_check_matrices with XOR-merged priors); Relay-BP decodes the exact syndromes (5708faf).
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import stim

# ECCentric's settings are "current"
VARIANTS = {
    "bposd": {"current": dict(max_iter=10000, ms_scaling_factor=1.0),
              "iter1000": dict(max_iter=1000, ms_scaling_factor=1.0),
              "iter100": dict(max_iter=100, ms_scaling_factor=1.0),
              "iter100_ms0.625": dict(max_iter=100, ms_scaling_factor=0.625)},
    "bplsd": {"current": dict(max_iter=10000, lsd_order=7),
              "iter100": dict(max_iter=100, lsd_order=7),
              "iter100_order10": dict(max_iter=100, lsd_order=10)},
    "belief_find": {"current": dict(max_iter=10000),
                    "iter1000": dict(max_iter=1000),
                    "iter100": dict(max_iter=100)},
    "relay_bp": {"current": dict(precision="f64", pre_iter=60, stop_nconv=5),
                 "decoder_bench": dict(precision="f32", pre_iter=80, stop_nconv=3)},
}
BUDGET_S = 1800
TIMEOUT_S = 3000
SEED = 20261105
DECODERS = tuple(VARIANTS)


def _circuit_id(job_id):
    return re.sub(r"\.decoder=.*$", "", re.sub(r"^decode\.", "circuit.", job_id))


def make(run_dir, results_csv, base, circuits=60):
    """Pick a stratified sample of measured sweep circuits with best LER 2e-3 to 0.3 and write the jobs"""
    import pandas as pd
    df = pd.read_csv(results_csv, low_memory=False)
    df = df[df.status.isin(["measured", "low statistics"]) & df.result_file.notna() & (df.shots > 0)
            & ~df.run.isin(["sweep_refine"]) & df.decoder.isin(DECODERS + ("tesseract", "mwpm"))]
    df = df.assign(ler=df.errors / df.shots, circuit=[_circuit_id(j) for j in df.job_id])
    per = df.groupby(["run", "circuit"]).agg(code=("code", "first"), backend=("backend", "first"), best=("ler", "min"),
                                             detectors=("detectors", "first"), slice=("slice", "first"))
    per = per[(per.best >= 2e-3) & (per.best <= 0.3) & (per.detectors <= 3000)].reset_index()
    per["family"] = per.code.str.extract(r"^([a-z_]+?)(?:\d|$)")[0]
    rng = np.random.default_rng(SEED)
    per = per.iloc[rng.permutation(len(per))]
    # One circuit per code and device, then round-robin over code families so every family is represented
    per = per.drop_duplicates(["code", "backend"])
    picked = []
    groups = [list(g.index) for _, g in per.groupby("family")]
    while len(picked) < circuits and any(groups):
        for g in groups:
            if g and len(picked) < circuits:
                picked.append(g.pop(0))
    sample = per.loc[picked]
    for sub in ("jobs", "claims", "results", "logs", "shots", "outcomes"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    count = 0
    for r in sample.itertuples():
        path = os.path.join(base, r.run, "circuits", f"{r.circuit}.stim")
        shots = int(np.clip(400 / r.best, 2000, 30000))
        sid = f"settings_shots.{r.run}.{r.circuit}"
        common = dict(module="sweep.run_job", circuit_file=path, code=r.code, backend=r.backend, family=r.family,
                      slice=r.slice, best_ler=float(r.best), detectors=int(r.detectors))
        jobs = [dict(id=sid, type="settings_shots", priority=0, shots=shots, timeout=1800, **common)]
        for decoder, variants in VARIANTS.items():
            for name in variants:
                jobs.append(dict(id=f"settings_decode.{r.run}.{r.circuit}.decoder={decoder}.variant={name}",
                                 type="settings_decode", priority=1, needs=sid, decoder=decoder, variant=name,
                                 budget_s=BUDGET_S, timeout=TIMEOUT_S, **common))
        for job in jobs:
            with open(os.path.join(run_dir, "jobs", f"{job['id']}.json"), "w") as f:
                json.dump(job, f)
            count += 1
    print(f"{len(sample)} circuits ({sample.family.nunique()} families), {count} jobs in {run_dir}")
    print(sample.family.value_counts().to_dict())


def _shots_path(run_dir, job):
    return os.path.join(run_dir, "shots", f"{job.get('needs', job['id'])}.npz")


def shots_job(job, run_dir):
    circuit = stim.Circuit.from_file(job["circuit_file"])
    dets, obs = circuit.compile_detector_sampler(seed=SEED).sample(job["shots"], separate_observables=True)
    path = _shots_path(run_dir, job)
    np.savez_compressed(path + ".tmp.npz", dets=np.packbits(dets, axis=1), obs=np.packbits(obs, axis=1),
                        num_detectors=dets.shape[1], num_observables=obs.shape[1])
    os.replace(path + ".tmp.npz", path)
    return {"shots": job["shots"], "num_detectors": int(dets.shape[1])}


def _load_shots(run_dir, job):
    z = np.load(_shots_path(run_dir, job))
    dets = np.unpackbits(z["dets"], axis=1, count=int(z["num_detectors"])).astype(np.uint8)
    obs = np.unpackbits(z["obs"], axis=1, count=int(z["num_observables"])).astype(bool)
    return dets, obs


def _ldpc_decoder(decoder, chk, priors, settings):
    from ldpc import BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder
    common = dict(error_channel=list(priors), max_iter=settings["max_iter"], bp_method="minimum_sum",
                  ms_scaling_factor=settings.get("ms_scaling_factor", 1.0))
    order = min(7, chk.shape[1] - chk.shape[0])
    if decoder == "bposd":
        return BpOsdDecoder(chk, osd_method="osd_cs", osd_order=order, **common)
    if decoder == "bplsd":
        return BpLsdDecoder(chk, lsd_method="lsd_cs", lsd_order=min(settings["lsd_order"], chk.shape[1] - chk.shape[0]),
                            **common)
    return BeliefFindDecoder(chk, uf_method="inversion", **common)


def decode_job(job, run_dir):
    from eccentric.decoders.bp_osd import dem_to_check_matrices
    circuit = stim.Circuit.from_file(job["circuit_file"])
    dets, obs = _load_shots(run_dir, job)
    settings = VARIANTS[job["decoder"]][job["variant"]]
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)
    c0 = time.process_time()
    wrong = np.zeros(len(dets), dtype=bool)
    if job["decoder"] == "relay_bp":
        import relay_bp
        from relay_bp.stim.sinter.check_matrices import CheckMatrices
        m = CheckMatrices.from_dem(dem)
        cls = relay_bp.RelayDecoderF32 if settings["precision"] == "f32" else relay_bp.RelayDecoderF64
        decoder = cls(m.check_matrix, error_priors=m.error_priors, gamma0=0.1, pre_iter=settings["pre_iter"], num_sets=60,
                      set_max_iter=60, gamma_dist_interval=(-0.24, 0.66), stop_nconv=settings["stop_nconv"])
        runner = relay_bp.ObservableDecoderRunner(decoder, m.observables_matrix, include_decode_result=False)
        build = time.process_time() - c0
        c1, done = time.process_time(), 0
        while done < len(dets) and time.process_time() - c1 < job["budget_s"]:
            chunk = slice(done, min(done + 100, len(dets)))
            s = dets[chunk]
            if m.syndrome_bias is not None:
                s = (s + m.syndrome_bias) % 2
            pred = runner.decode_observables_batch(s, parallel=False).astype(bool)
            if m.observables_bias is not None:
                pred = pred ^ m.observables_bias.astype(bool)
            wrong[chunk] = np.any(pred != obs[chunk], axis=1)
            done = chunk.stop
    else:
        chk, om, priors = dem_to_check_matrices(dem, xor_priors=True)
        if job["decoder"] == "belief_find":
            seen = np.asarray(chk.sum(axis=0)).ravel() > 0
            chk, om, priors = chk[:, seen], om[:, seen], np.asarray(priors)[seen]
        decoder = _ldpc_decoder(job["decoder"], chk, priors, settings)
        build = time.process_time() - c0
        c1, done = time.process_time(), 0
        while done < len(dets) and time.process_time() - c1 < job["budget_s"]:
            for i in range(done, min(done + 100, len(dets))):
                wrong[i] = np.any((om @ decoder.decode(dets[i])) % 2 != obs[i])
            done = min(done + 100, len(dets))
    cpu = time.process_time() - c1
    np.save(os.path.join(run_dir, "outcomes", f"{job['id']}.npy"), np.packbits(wrong[:done]))
    return {"shots": int(done), "errors": int(wrong[:done].sum()), "decode_cpu_s": cpu, "build_cpu_s": build,
            "ms_per_shot": 1000 * cpu / max(done, 1), "complete": bool(done == len(dets))}


def run(job, run_dir):
    return shots_job(job, run_dir) if job["type"] == "settings_shots" else decode_job(job, run_dir)


def _mcnemar_p(b, c):
    """Two-sided exact McNemar p value for b and c discordant shots"""
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue


def report(run_dir, out_csv):
    """Per circuit, decoder and variant: paired with the current setting on their common shots"""
    import pandas as pd
    rows = []
    for name in os.listdir(os.path.join(run_dir, "results")):
        if not name.startswith("settings_decode.") or not name.endswith(".json"):
            continue
        r = json.load(open(os.path.join(run_dir, "results", name)))
        if r.get("status") != "done":
            rows.append({**{k: r["job"][k] for k in ("id", "code", "family", "backend", "decoder", "variant", "needs")},
                         "status": r.get("status"), "error": r.get("error")})
            continue
        rows.append({**{k: r["job"][k] for k in ("id", "code", "family", "backend", "decoder", "variant", "needs",
                                                  "best_ler", "detectors")},
                     "status": "done", **{k: r[k] for k in ("shots", "errors", "ms_per_shot", "build_cpu_s", "complete")}})
    df = pd.DataFrame(rows)
    out = []
    for (needs, decoder), g in df[df.status == "done"].groupby(["needs", "decoder"]):
        cur = g[g.variant == "current"]
        if cur.empty:
            continue
        cur = cur.iloc[0]
        wc = np.unpackbits(np.load(os.path.join(run_dir, "outcomes", f"{cur['id']}.npy")))[:cur.shots].astype(bool)
        for v in g.itertuples():
            wv = np.unpackbits(np.load(os.path.join(run_dir, "outcomes", f"{v.id}.npy")))[:v.shots].astype(bool)
            n = min(len(wc), len(wv))
            b, c = int((wv[:n] & ~wc[:n]).sum()), int((~wv[:n] & wc[:n]).sum())
            out.append({"circuit": needs.split(".", 1)[1], "code": v.code, "family": v.family, "backend": v.backend,
                        "detectors": v.detectors, "decoder": decoder, "variant": v.variant, "common_shots": n,
                        "errors_current": int(wc[:n].sum()), "errors_variant": int(wv[:n].sum()),
                        "ler_current": wc[:n].mean(), "ler_variant": wv[:n].mean(),
                        "ler_ratio": (wv[:n].sum() + 0.5) / (wc[:n].sum() + 0.5), "worse_only": b, "better_only": c,
                        "p_mcnemar": _mcnemar_p(b, c), "ms_current": cur.ms_per_shot, "ms_variant": v.ms_per_shot,
                        "time_ratio": v.ms_per_shot / cur.ms_per_shot if cur.ms_per_shot else math.nan,
                        "build_s_variant": v.build_cpu_s})
    table = pd.DataFrame(out)
    table.to_csv(out_csv, index=False)
    failed = df[df.status != "done"]
    print(f"{len(table)} paired rows, {table.circuit.nunique() if len(table) else 0} circuits; not done: {len(failed)}")
    if len(failed):
        print(failed.groupby(["decoder", "variant", "status"]).size().to_string())
    if len(table):
        s = table[table.variant != "current"].groupby(["decoder", "variant"]).agg(
            circuits=("circuit", "nunique"), median_ler_ratio=("ler_ratio", "median"),
            significantly_worse=("p_mcnemar", lambda p: 0), median_time_ratio=("time_ratio", "median"))
        worse = table[(table.p_mcnemar < 0.01) & (table.worse_only > table.better_only)].groupby(["decoder", "variant"]).size()
        better = table[(table.p_mcnemar < 0.01) & (table.better_only > table.worse_only)].groupby(["decoder", "variant"]).size()
        s["significantly_worse"] = worse.reindex(s.index).fillna(0).astype(int)
        s["significantly_better"] = better.reindex(s.index).fillna(0).astype(int)
        print(s.round(3).to_string())


def main():
    if sys.argv[1] == "make":
        make(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 60)
    elif sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
