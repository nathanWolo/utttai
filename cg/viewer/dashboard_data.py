"""Parse the experiment logs under utttai/ and utttai/train/ into one JSON document
for the dashboard. Results are cached per file on (size, mtime), so polling is cheap.
"""
import json
import math
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = ROOT / "train"
RUNNING_WINDOW = 150  # a log written to within this many seconds counts as a running job

_cache = {}


def cached(path, parse):
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), parse.__name__)
    stamp = (st.st_size, st.st_mtime)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        value = parse(path)
    except Exception as e:  # a half-written log must not break the page
        value = {"error": str(e)}
    _cache[key] = (stamp, value)
    return value


def lines(path):
    return path.read_text(encoding="utf-8", errors="replace").replace("\r", "").splitlines()


def age(path):
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 1e9


# ---------------------------------------------------------------- CF vs uttt.ai matches
PROG = re.compile(r"^\[(\d\d:\d\d:\d\d)\] (\d+)/(\d+) games  ETA (\S+)  crossfish W/D/L uttt\.ai rules \((\d+), (\d+), (\d+)\) "
                  r"Elo ([+-]?\d+) \+/- (\d+) \| CodinGame rules \((\d+), (\d+), (\d+)\) Elo ([+-]?\d+) \+/- (\d+)")
FINAL = re.compile(r"^(uttt\.ai rules|CodinGame rules) *: crossfish W/D/L \((\d+), (\d+), (\d+)\)  Elo ([+-]?[\d.]+) \+/- ([\d.]+)")


def parse_match(path):
    ls = lines(path)
    if not ls:
        return None
    head = ls[0]
    cf_ms = re.search(r"crossfish (\d+) ms/move", head)
    ut = re.search(r"uttt\.ai (\d+)(ms| simulations)/move", head)
    net = re.search(r"net (\S+?):", head)
    games = re.search(r"= (\d+) games", head)
    m = dict(name=path.stem, file=str(path.relative_to(ROOT)), header=head,
             cf_ms=int(cf_ms.group(1)) if cf_ms else None,
             ut_ms=int(ut.group(1)) if ut and ut.group(2) == "ms" else None,
             ut_sims=int(ut.group(1)) if ut and ut.group(2) != "ms" else None,
             net=Path(net.group(1)).stem if net else ("original" if "rules module src:" in head or "module src," in head else None),
             exe=head.split(" ")[0], total=int(games.group(1)) if games else None,
             progress=[], final={}, done=0)
    if m["net"] is None and "net " not in head:
        m["net"] = "original"
    for l in ls[1:]:
        p = PROG.match(l)
        if p:
            g = p.groups()
            m["progress"].append(dict(t=g[0], k=int(g[1]), ut_wdl=[int(x) for x in g[4:7]], ut_elo=int(g[7]), ut_ci=int(g[8]),
                                      cg_wdl=[int(x) for x in g[9:12]], cg_elo=int(g[12]), cg_ci=int(g[13])))
            m["done"], m["eta"] = int(g[1]), g[3]
        f = FINAL.match(l)
        if f:
            key = "ut" if f.group(1).startswith("uttt") else "cg"
            m["final"][key] = dict(wdl=[int(f.group(i)) for i in (2, 3, 4)], elo=float(f.group(5)), ci=float(f.group(6)))
        if l.startswith("full board, no line"):
            c = re.search(r"(\d+)/(\d+)", l)
            m["count_endings"] = [int(c.group(1)), int(c.group(2))]
        if l.startswith("mean think time"):
            c = re.search(r"crossfish ([\d.]+) ms, uttt\.ai ([\d.]+) s", l)
            if c:
                m["think"] = dict(cf_ms=float(c.group(1)), ut_ms=float(c.group(2)) * 1000)
    m["finished"] = bool(m["final"])
    return m


def parse_live(live):
    """Per-move metadata from a match's live feed: CF depth, uttt.ai simulations, plies."""
    cf_d, ut_s, plies, n = [], [], [], 0
    cf_nps, ut_sps = [], []
    for f in live.glob("g*.json"):
        try:
            g = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        n += 1
        plies.append(len(g["opening"]) + len(g["moves"]))
        for mv in g["moves"]:
            if mv["by"] == "cf" and "d" in mv and not mv.get("book") and abs(mv.get("e", 0)) < 20000:
                cf_d.append(mv["d"])
                if mv.get("ms", 0) >= 20 and "n" in mv:
                    cf_nps.append(mv["n"] / mv["ms"] * 1000)
            if mv["by"] == "ut" and "s" in mv:
                ut_s.append(mv["s"])
                if mv.get("ms", 0) >= 20 and mv["s"] < 50 * mv["ms"]:  # skip near-solved trees
                    ut_sps.append(mv["s"] / mv["ms"] * 1000)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    # medians: end-game searches over nearly solved trees run thousands of cheap simulations
    return dict(games=n, cf_depth=med(cf_d), ut_sims=med(ut_s), plies=med(plies),
                cf_nps=med(cf_nps), ut_sps=med(ut_sps))


def live_stats(live):
    # the directory mtime does not change when files are replaced in place, so key on the newest file
    try:
        newest = max((f.stat().st_mtime for f in live.glob("*.json")), default=0)
    except OSError:
        newest = 0
    key = (str(live), "live")
    hit = _cache.get(key)
    if hit and (hit[0] == newest or time.time() - hit[2] < 20):
        return hit[1]
    value = parse_live(live)
    _cache[key] = (newest, value, time.time())
    return value


def matches():
    out = []
    for path in list(ROOT.glob("*.log")) + list(TRAIN.glob("*.log")):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                head = fh.readline()
        except OSError:
            continue
        if "ms/move vs uttt.ai" not in head or path.stem.startswith("smoke"):
            continue
        m = cached(path, parse_match)
        if not m or "error" in m:
            continue
        m = dict(m)
        m["running"] = not m["finished"] and age(path) < RUNNING_WINDOW
        m["mtime"] = path.stat().st_mtime
        live = path.with_suffix(".live")
        if live.is_dir():
            m["live"] = live.name
            m["meta"] = live_stats(live)
        out.append(m)
    return sorted(out, key=lambda m: m["mtime"], reverse=True)


# ---------------------------------------------------------------- training
def parse_gate_sprt(path):
    pts, verdict, head = [], None, ""
    for l in lines(path):
        if l.startswith("SPRT A="):
            head = l
        r = re.search(r"\] (\d+) pairs .*Elo\(A\) ([+-][\d.]+) \+/- ([\d.]+)  LLR ([+-][\d.]+)", l)
        if r:
            pts.append([int(r.group(1)), float(r.group(4)), float(r.group(2)), float(r.group(3))])
        if l.startswith("SPRT H") or l.startswith("SPRT inconclusive"):
            verdict = l.strip()
    a = re.search(r"A=(\S+) vs B=(\S+);", head)
    return dict(name=path.stem, a=a.group(1) if a else None, b=a.group(2) if a else None, points=pts, verdict=verdict)


def parse_fixed_gate(path):
    last = None
    head = lines(path)[0] if path.exists() else ""
    for l in lines(path):
        r = re.search(r"(\d+)/(\d+) games .*A W/D/L (\d+)/(\d+)/(\d+)  Elo\(A\) ([+-][\d.]+) \+/- ([\d.]+)", l)
        if r:
            last = dict(games=int(r.group(1)), total=int(r.group(2)), wdl=[int(r.group(i)) for i in (3, 4, 5)],
                        elo=float(r.group(6)), ci=float(r.group(7)))
    a = re.search(r"A=(\S+) vs B=(\S+);", head)
    return dict(result=last, a=a.group(1) if a else None, b=a.group(2) if a else None)


def parse_train(path):
    for l in reversed(lines(path)):
        r = re.search(r"validation after epoch (\d+): policy CE ([\d.]+)  anchor CE ([\d.]+)  value MSE ([\d.]+)  MSE vs game result ([\d.]+)", l)
        if r:
            return dict(epoch=int(r.group(1)), policy_ce=float(r.group(2)), anchor_ce=float(r.group(3)),
                        value_mse=float(r.group(4)), result_mse=float(r.group(5)))
    return None


def parse_selfplay(path):
    last = None
    for l in lines(path):
        r = re.search(r"^\[(\d\d:\d\d:\d\d)\] (\d+)/(\d+) games  ([\d,]+) games/h  ETA (\S+)", l)
        if r:
            last = dict(t=r.group(1), done=int(r.group(2)), total=int(r.group(3)), rate=int(r.group(4).replace(",", "")), eta=r.group(5))
    return last


def training():
    gens = []
    # net1 was fine-tuned before the loop existed; its gate was a fixed 400-game match vs the original
    n1 = TRAIN / "match_net1_vs_orig.log"
    if n1.exists():
        r = cached(n1, parse_fixed_gate)["result"]
        if r:
            gens.append(dict(net="net1", vs="original", elo=r["elo"], ci=r["ci"], games=r["games"], method="fixed 400 games",
                             promoted=True))
    loop = TRAIN / "loop.log"
    decisions = {}
    if loop.exists():
        for l in lines(loop):
            r = re.search(r"(ACCEPTED|rejected) (net\d+): ([+-][\d.]+) \+/- ([\d.]+)", l)
            if r:
                decisions[r.group(2)] = dict(promoted=r.group(1) == "ACCEPTED", method="fixed 400 games")
            r = re.search(r"(PROMOTED|not promoted) (net\d+)", l)
            if r:
                decisions[r.group(2)] = dict(promoted=r.group(1) == "PROMOTED", method="SPRT")
            r = re.search(r"handover: net3 vs net2 SPRT (H\d) accepted", l)
            if r:
                decisions["net3"] = dict(promoted=r.group(1) == "H1", method="SPRT (completion-order, biased)")
            if "stopped by request during the net5 SPRT" in l:
                decisions.setdefault("net5", dict(promoted=False, method="SPRT (stopped, inconclusive)"))
    for k in range(2, 20):
        net = f"net{k}"
        if not (TRAIN / f"{net}.pt").exists():
            continue
        d = decisions.get(net, dict(promoted=None, method="?"))
        entry = dict(net=net, promoted=d["promoted"], method=d["method"])
        sprt_path = TRAIN / (f"gate{k}.log")
        sprt = cached(sprt_path, parse_gate_sprt) if sprt_path.exists() else None
        if sprt and sprt["points"]:
            p = sprt["points"][-1]
            entry.update(vs=sprt["b"].replace(".pt", ""), elo=p[2], ci=p[3], llr=p[1], pairs=p[0], sprt=sprt["verdict"])
        else:
            fg = cached(sprt_path, parse_fixed_gate) if sprt_path.exists() else None
            if fg and fg["result"]:
                entry.update(vs=fg["b"].replace(".pt", ""), elo=fg["result"]["elo"], ci=fg["result"]["ci"], games=fg["result"]["games"])
        if net == "net3":  # its fixed gate (unbiased) is the estimate used for the ladder
            entry["note"] = "fixed gate +16.5; biased SPRT +32.9 promoted it"
        gens.append(entry)
    for g in gens:
        t = TRAIN / f"train{g['net'][3:]}.log"
        g["val"] = cached(t, parse_train) if t.exists() else None
        sp = TRAIN / f"gen{int(g['net'][3:])}.log"
        g["selfplay"] = cached(sp, parse_selfplay) if sp.exists() else None
    sprts = []
    for path in sorted(TRAIN.glob("gate*.log")) + [TRAIN / "sprt3.log"]:
        if not path.exists():
            continue
        s = cached(path, parse_gate_sprt)
        if s and s["points"]:
            s = dict(s)
            s["biased"] = path.name == "sprt3.log"
            s["running"] = s["verdict"] is None and age(path) < RUNNING_WINDOW
            sprts.append(s)
    return dict(generations=gens, sprts=sprts)


# ---------------------------------------------------------------- activity
def activity(ms):
    jobs = []
    for m in ms:
        if m["running"]:
            last = m["progress"][-1] if m["progress"] else None
            jobs.append(dict(kind="match", title=m["name"], done=m["done"], total=m["total"], eta=m.get("eta"),
                             detail=(f"CF {last['cg_elo']:+d} ± {last['cg_ci']} Elo" if last else "starting")))
    for path in TRAIN.glob("gen*.log"):
        if age(path) < RUNNING_WINDOW:
            sp = cached(path, parse_selfplay)
            if sp and sp["done"] < sp["total"]:
                jobs.append(dict(kind="selfplay", title=path.stem, done=sp["done"], total=sp["total"], eta=sp["eta"],
                                 detail=f"{sp['rate']:,} games/h"))
    for path in TRAIN.glob("gate*.log"):
        if age(path) < RUNNING_WINDOW:
            s = cached(path, parse_gate_sprt)
            if s and s["points"] and not s["verdict"]:
                p = s["points"][-1]
                jobs.append(dict(kind="sprt", title=f"{s['a']} vs {s['b']}", done=None, total=None, eta=None,
                                 detail=f"{p[0]} pairs · Elo {p[2]:+.1f} ± {p[3]:.1f} · LLR {p[1]:+.2f}"))
    plan = TRAIN / "bench_net4.log"
    queue = []
    if plan.exists() and age(plan) < 6 * 3600:
        text = plan.read_text(encoding="utf-8", errors="replace")
        if "=== done" not in text:
            started = set(re.findall(r"start: (\d+) ms per move", text))
            queue = [f"{ms} ms" for ms in ("90", "1000", "5000") if ms not in started]
    return dict(jobs=jobs, queue=queue)


def dashboard():
    ms = matches()
    return dict(now=time.time(), matches=ms, training=training(), activity=activity(ms))


if __name__ == "__main__":
    print(json.dumps(dashboard(), indent=1)[:4000])
