"""Slice results of tcbp_jobs_gpu (results/) and tcbp_jobs_sc_gpu (results_tcbp_jobs_sc_gpu/) against each other and the
CPU decoders of tools/matchmaker/data/dataset.csv (95% Wilson intervals); circuits in mismatched.json (slice.json
named another run's circuit of the same name) are excluded. python analyze.py <dir with tcbp_res, tcbp_sc, mismatched.json,
runner.log, runner_sc.log>"""
import csv, json, glob, collections, math, sys, os
S = sys.argv[1]
def wilson(e, n, z=1.96):
    if n == 0: return (0, 1)
    p = e / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0, c-h), c+h
D = collections.defaultdict(dict)
for r in csv.DictReader(open('tools/matchmaker/data/dataset.csv')):
    if r['shots'] and r['errors'] and float(r['shots']) > 0:
        D[r['circuit_key']][r['decoder']] = dict(e=int(float(r['errors'])), n=int(float(r['shots'])), status=r['status'], ms=float(r['ms_per_shot'] or 'nan'))
def load(d):
    out = {}
    for f in glob.glob(f"{S}/{d}/*.json"):
        o = json.load(open(f)); out[o['index']] = o
    return out
M, SC = load("tcbp_res"), load("tcbp_sc")
EXCL = set(b[0] for b in json.load(open(S + "/mismatched.json")))
M = {i: o for i, o in M.items() if i not in EXCL}; SC = {i: o for i, o in SC.items() if i not in EXCL}
bad = [i for i, o in M.items() if 'error' in o.get('ler', {}) or 'error' in o.get('timing', {}) or 'error' in o]
print("main circuits", len(M), "sc", len(SC), "with errors:", bad, [M[i].get('ler', {}).get('error', M[i].get('timing', {}).get('error')) for i in bad][:3])
def cmp(a, b):
    """-1 a significantly lower LER than b, +1 higher, 0 overlapping 95% intervals"""
    la, ha = wilson(a['e'], a['n']); lb, hb = wilson(b['e'], b['n'])
    return -1 if ha < lb else (1 if la > hb else 0)
# 1. sc vs main
c = collections.Counter(); ratios = []; tr = []
for i in SC:
    if i not in M: continue
    a = dict(e=SC[i]['ler']['errors'], n=SC[i]['ler']['shots']); b = dict(e=M[i]['ler']['errors'], n=M[i]['ler']['shots'])
    c[cmp(a, b)] += 1
    if a['e'] >= 10 and b['e'] >= 10: ratios.append((a['e']/a['n']) / (b['e']/b['n']))
    tr.append(SC[i]['timing']['gpu_ms_per_shot'] / M[i]['timing']['gpu_ms_per_shot'])
med = lambda x: sorted(x)[len(x)//2] if x else float('nan')
print(f"\nsc vs main ({sum(c.values())} circuits): sc lower {c[-1]}, same {c[0]}, sc higher {c[1]}; LER ratio sc/main median {med(ratios):.3f}, min {min(ratios):.3f}; GPU time ratio median {med(tr):.2f}, p90 {sorted(tr)[int(.9*len(tr))]:.2f}, max {max(tr):.2f}")
worst = sorted(((r, i) for i, r in ((i, SC[i]['timing']['gpu_ms_per_shot']/M[i]['timing']['gpu_ms_per_shot']) for i in SC if i in M)), reverse=True)[:3]
print("  slowest sc vs main:", [(i, M[i]['code'], round(r, 2)) for r, i in worst])
# 2. each vs CPU decoders
for name, R in (("tcbp_jobs_gpu", M), ("tcbp_jobs_sc_gpu", SC)):
    print(f"\n{name} vs CPU decoders (95% Wilson intervals; 'measured' CPU rows only)")
    for dec in ("tesseract", "bposd_tuned", "relay_bp", "bposd", "bplsd", "mwpm", "mwpm_corr", "chromobius", "mwpf"):
        cc = collections.Counter(); rs = []
        for i, o in R.items():
            cpu = D.get(o['key'], {}).get(dec)
            if not cpu or cpu['status'] != 'measured': continue
            a = dict(e=o['ler']['errors'], n=o['ler']['shots'])
            cc[cmp(a, cpu)] += 1
            if a['e'] >= 10 and cpu['e'] >= 10: rs.append((a['e']/a['n']) / (cpu['e']/cpu['n']))
        if cc: print(f"  vs {dec:12s} n={sum(cc.values()):3d}: tcbp better {cc[-1]:3d}, same {cc[0]:3d}, worse {cc[1]:3d}; LER ratio median {med(rs):.2f}")
    # vs best CPU decoder measured on the circuit
    cc = collections.Counter(); worse = []
    for i, o in R.items():
        cpus = {d: v for d, v in D.get(o['key'], {}).items() if v['status'] == 'measured'}
        if not cpus: continue
        bd, bv = min(cpus.items(), key=lambda kv: kv[1]['e']/kv[1]['n'])
        a = dict(e=o['ler']['errors'], n=o['ler']['shots']); r = cmp(a, bv); cc[r] += 1
        if r == 1: worse.append((i, o['code'], o['key'].split('backend=')[1].split('.')[0], bd, f"{a['e']}/{a['n']}", f"{bv['e']}/{bv['n']}"))
    print(f"  vs best CPU decoder per circuit n={sum(cc.values())}: better {cc[-1]}, same {cc[0]}, worse {cc[1]}")
    for w in worse[:12]: print("    worse:", w)
# 3. GPU time used
def wall(log):
    st = {}; tot = 0
    for line in open(log):
        p = line.split()
        if len(p) >= 3 and p[1] == 'start': st[p[2]] = p[0]
        elif len(p) >= 3 and p[1] == 'end' and p[2] in st:
            from datetime import datetime
            f = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
            tot += (f(p[0]) - f(st[p[2]])).total_seconds()
    return tot / 3600
print(f"\nGPU wall hours: main {wall(S+'/runner.log'):.1f} (306), sc {wall(S+'/runner_sc.log'):.1f} (293)")
