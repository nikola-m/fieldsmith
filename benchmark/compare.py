"""Run both benchmarks and print the side-by-side comparison.

    python compare.py 5                      # 5 runs per task in the warm/first-run scenario
    python compare.py 5 --md results.md      # also write a Markdown report (and results.json next to it)

Needs: python with sympy, mpmath and fieldsmith (see bench_fieldsmith.py), and julia on the PATH with this
folder's project instantiated (see bench_alakazam.jl). Takes several minutes; the one-off scenario starts
36 fresh processes.

Scenarios
  warm      best of runs 2..N of each task, all tasks in one process per tool
  first     run 1 of each task in that same process (Julia: includes JIT compilation of whatever that
            task is first to use, so it depends on task order)
  one-off   each task alone in a fresh process, timed from process start to exit (startup, package load,
            compilation and the task), i.e. what running a single example script costs
  suite     all 18 tasks once in one fresh process, timed from start to exit
  load      a fresh process that only loads the packages
"""
import json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
md_path = None
if "--md" in args:
    md_path = args[args.index("--md") + 1]
    del args[args.index("--md"):args.index("--md") + 2]
reps = args[0] if args else "5"

PY = [sys.executable, "bench_fieldsmith.py"]
JL = ["julia", "--project=.", "bench_alakazam.jl"]


def run(cmd):
    t = time.perf_counter()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t
    rows = [json.loads(x) for x in re.findall(r"^RESULT (.*)$", p.stdout, re.M)]
    if p.returncode != 0 or not rows:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return {r["task"]: r for r in rows}, wall


def wall_of(cmd):
    t = time.perf_counter()
    subprocess.run(cmd, cwd=HERE, capture_output=True, check=True)
    return time.perf_counter() - t


def norm(s):
    # Put both tools' printed results in one notation: SymPy prints m**2 and mu/2 where Julia prints
    # m^2 and (1//2)*mu (the Chern-Simons coefficient); fit dicts are compared with their keys sorted.
    s = s.strip().lower().replace("'", "").replace("**", "^").replace("(1//2)*mu", "mu/2")
    if s.startswith("{"):
        s = "{" + ", ".join(sorted(x.strip() for x in s[1:-1].split(","))) + "}"
    return s


log = lambda m: print(m, file=sys.stderr, flush=True)
log("warm and first-run timings ...")
fs, _ = run(PY + [reps, "--json"])
jl, _ = run(JL + [reps, "--json"])
tasks = list(fs)
log("whole suite once ...")
_, fs_suite = run(PY + ["1", "--json"])
_, jl_suite = run(JL + ["1", "--json"])
log("package load only ...")
fs_load = wall_of([sys.executable, "-c", "import os, sys; p = os.environ.get('FIELDSMITH_PATH'); p and sys.path.insert(0, p); "
                   "import sympy, fieldsmith, fieldsmith.gauge, fieldsmith.noether, fieldsmith.analysis"])
jl_load = wall_of(["julia", "--project=.", "-e", "using Alakazam, Symbolics"])
fs_one, jl_one = {}, {}
for k, t in enumerate(tasks, 1):
    log(f"one-off {k}/{len(tasks)}: {t}")
    fs_one[t] = run(PY + ["1", "--json", f"--only={k}"])[1]
    jl_one[t] = run(JL + ["1", "--json", f"--only={k}"])[1]

rows = []
for t in tasks:
    f, j = fs[t], jl[t]
    rows.append(dict(task=t, result=f["result"], same=norm(f["result"]) == norm(j["result"]),
                     fs_warm=min(f["runs"][1:]), jl_warm=min(j["runs"][1:]),
                     fs_first=f["runs"][0], jl_first=j["runs"][0],
                     fs_one=fs_one[t], jl_one=jl_one[t]))


def fmt_ms(x):
    return f"{x * 1e3:,.1f}" if x < 1 else f"{x * 1e3:,.0f}"


def ratio_str(f, j):
    r = f / j
    return f"{r:.0f}x" if r >= 10 else (f"{r:.1f}x" if r >= 1 else f"1/{1 / r:.1f}x")


warm_r = [r["fs_warm"] / r["jl_warm"] for r in rows]
first_fs_faster = sum(r["fs_first"] < r["jl_first"] for r in rows)
one_fs_faster = sum(r["fs_one"] < r["jl_one"] for r in rows)
tot = {k: sum(r[k] for r in rows) for k in ("fs_warm", "jl_warm", "fs_first", "jl_first", "fs_one", "jl_one")}

# ---------------------------------------------------------------- plain text
hdr = f"{'task':40s}{'warm fs':>10s}{'warm az':>10s}{'ratio':>8s}{'first fs':>10s}{'first az':>10s}{'ratio':>8s}{'one-off fs':>12s}{'one-off az':>12s}{'ratio':>8s}  same"
print("\ntimes in ms; ratio = fieldsmith time / Alakazam time (1/k x: fieldsmith k times faster)\n")
print(hdr)
for r in rows:
    print(f"{r['task'][:39]:40s}{fmt_ms(r['fs_warm']):>10s}{fmt_ms(r['jl_warm']):>10s}{ratio_str(r['fs_warm'], r['jl_warm']):>8s}"
          f"{fmt_ms(r['fs_first']):>10s}{fmt_ms(r['jl_first']):>10s}{ratio_str(r['fs_first'], r['jl_first']):>8s}"
          f"{fmt_ms(r['fs_one']):>12s}{fmt_ms(r['jl_one']):>12s}{ratio_str(r['fs_one'], r['jl_one']):>8s}  {'yes' if r['same'] else 'NO'}")
print(f"{'TOTAL':40s}{fmt_ms(tot['fs_warm']):>10s}{fmt_ms(tot['jl_warm']):>10s}{ratio_str(tot['fs_warm'], tot['jl_warm']):>8s}"
      f"{fmt_ms(tot['fs_first']):>10s}{fmt_ms(tot['jl_first']):>10s}{ratio_str(tot['fs_first'], tot['jl_first']):>8s}"
      f"{fmt_ms(tot['fs_one']):>12s}{fmt_ms(tot['jl_one']):>12s}{ratio_str(tot['fs_one'], tot['jl_one']):>8s}")
summary = [
    f"warm:     per-task ratio median {statistics.median(warm_r):.0f}x, range {min(warm_r):.0f}x - {max(warm_r):.0f}x",
    f"first:    fieldsmith faster on {first_fs_faster} of {len(rows)} tasks (Julia's run 1 includes JIT and depends on task order)",
    f"one-off:  fieldsmith faster on {one_fs_faster} of {len(rows)} tasks (fresh process per task, incl. package load)",
    f"suite:    all {len(rows)} tasks once in one fresh process: fieldsmith {fs_suite:.1f} s, Alakazam {jl_suite:.1f} s",
    f"load:     package load only: fieldsmith {fs_load:.1f} s, Alakazam {jl_load:.1f} s",
    f"results:  {sum(r['same'] for r in rows)} of {len(rows)} final results agree",
]
print()
print("\n".join(summary))

# ---------------------------------------------------------------- markdown + json
if md_path:
    L = ["| task | result | warm fieldsmith | warm Alakazam | warm ratio | first fieldsmith | first Alakazam | first ratio | one-off fieldsmith | one-off Alakazam | one-off ratio |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['result']}{'' if r['same'] else ' (MISMATCH)'} | {fmt_ms(r['fs_warm'])} | {fmt_ms(r['jl_warm'])} | {ratio_str(r['fs_warm'], r['jl_warm'])} | "
                 f"{fmt_ms(r['fs_first'])} | {fmt_ms(r['jl_first'])} | {ratio_str(r['fs_first'], r['jl_first'])} | "
                 f"{fmt_ms(r['fs_one'])} | {fmt_ms(r['jl_one'])} | {ratio_str(r['fs_one'], r['jl_one'])} |")
    L.append(f"| **total** | | {fmt_ms(tot['fs_warm'])} | {fmt_ms(tot['jl_warm'])} | {ratio_str(tot['fs_warm'], tot['jl_warm'])} | "
             f"{fmt_ms(tot['fs_first'])} | {fmt_ms(tot['jl_first'])} | {ratio_str(tot['fs_first'], tot['jl_first'])} | "
             f"{fmt_ms(tot['fs_one'])} | {fmt_ms(tot['jl_one'])} | {ratio_str(tot['fs_one'], tot['jl_one'])} |")
    L += ["", "Times in milliseconds; ratio = fieldsmith time / Alakazam time (1/k x: fieldsmith k times faster).", ""]
    L += [f"- {s}" for s in summary]
    open(md_path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"rows": rows, "suite": {"fieldsmith": fs_suite, "alakazam": jl_suite},
               "load": {"fieldsmith": fs_load, "alakazam": jl_load}, "reps": reps},
              open(os.path.splitext(md_path)[0] + ".json", "w"), indent=1)
