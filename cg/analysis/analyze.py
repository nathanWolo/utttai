"""Mine a recorded crossfish vs uttt.ai match (a *.live directory written by
bench_vs_utttai.py) for how the engines play and judge positions differently.

Everything is computed from the logged moves and search metadata; no engine runs.
Writes <out>.json (for the report page) and prints a text summary.

usage: python analyze.py <match.live> <out.json>
"""
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # this repository's utttpy (CodinGame rules)
from utttpy.game.action import Action  # noqa: E402
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe  # noqa: E402

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
MATE = 90000  # crossfish mate scores are +/-(99999 - ply); CORR_MATE_BOUND in the engine


def load(live):
    games = []
    for f in sorted(glob.glob(str(Path(live) / "g*.json"))):
        g = json.load(open(f, encoding="utf-8"))
        if g.get("status") == "done":
            games.append(g)
    return games


def replay(g):
    """Per ply: the position before the move plus facts about the move."""
    u = UltimateTicTacToe()
    seq = g["opening"] + [m["i"] for m in g["moves"]]
    n_open = len(g["opening"])
    cf_sym = 1 if g["cf_is_x"] else 2
    plies = []
    for k, idx in enumerate(seq):
        mover = u.next_symbol
        sup_before = list(u.state[81:90])
        legal_before = u.get_legal_indexes()
        free_choice = len({i // 9 for i in legal_before}) > 1
        u.execute(Action(symbol=mover, index=idx), verify=False)
        sup_after = list(u.state[81:90])
        mb, sq = idx // 9, idx % 9
        captured = sup_before[mb] == 0 and sup_after[mb] == mover
        closed = sup_before[mb] == 0 and sup_after[mb] != 0
        gives_free = not u.is_terminated() and sup_after[sq] != 0  # opponent may play anywhere
        plies.append(dict(k=k, idx=idx, mover="cf" if mover == cf_sym else "ut", book=k < n_open,
                          meta=g["moves"][k - n_open] if k >= n_open else None, free_choice=free_choice,
                          captured=captured, closed=closed, gives_free=gives_free, n_legal=len(legal_before),
                          center=sq == 4, corner=sq in (0, 2, 6, 8), mb_center=mb == 4))
    sup = list(u.state[81:90])
    return plies, sup, cf_sym


def outcome_score(g):  # crossfish's score: 1 win, 0.5 draw, 0 loss (CodinGame rules)
    return (g["result"]["cg"] + 1) / 2


def logistic_fit(x, y, iters=300):
    """1-D logistic regression y ~ sigmoid(a*x + b) by Newton's method (y may be 0, .5, 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * x + b)))
        w = p * (1 - p) + 1e-9
        g_a, g_b = np.sum((y - p) * x), np.sum(y - p)
        h_aa, h_ab, h_bb = -np.sum(w * x * x), -np.sum(w * x), -np.sum(w)
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def brier(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def main():
    live, out_path = sys.argv[1], sys.argv[2]
    games = load(live)
    R = {"games": len(games), "source": Path(live).name}

    # ------------------------------------------------------------ outcomes
    ends = Counter()
    count_margin = Counter()
    by_color = defaultdict(Counter)
    lengths = defaultdict(list)
    rows = []
    for g in games:
        plies, sup, cf_sym = replay(g)
        g["_plies"], g["_sup"], g["_cf"] = plies, sup, cf_sym
        r = g["result"]["cg"]
        kind = "full board (count)" if g["result"]["count_ending"] else "three in a row"
        ends[(kind, r)] += 1
        by_color["CF as X" if g["cf_is_x"] else "CF as O"][r] += 1
        lengths[kind].append(len(plies))
        if g["result"]["count_ending"]:
            mine, theirs = sup.count(cf_sym), sup.count(3 - cf_sym)
            count_margin[(mine, theirs, sup.count(3))] += 1
    R["endings"] = [dict(kind=k, result=r, n=n) for (k, r), n in sorted(ends.items())]
    R["by_color"] = {k: [c[1], c[0], c[-1]] for k, c in by_color.items()}
    R["length"] = {k: dict(mean=float(np.mean(v)), median=float(np.median(v))) for k, v in lengths.items()}
    R["count_boards"] = [dict(cf=a, ut=b, drawn=c, n=n) for (a, b, c), n in count_margin.most_common(12)]

    # pairs: each opening played with both colours
    pairs = defaultdict(list)
    for g in games:
        pairs[g["id"] // 2].append(outcome_score(g))
    penta = Counter(sum(v) for v in pairs.values() if len(v) == 2)
    R["pentanomial"] = [penta.get(s, 0) for s in (0, 0.5, 1, 1.5, 2)]

    # ------------------------------------------------------------ evaluations vs outcome
    cf_x, cf_y, cf_ply, ut_x, ut_y, ut_ply = [], [], [], [], [], []
    for g in games:
        y = outcome_score(g)
        for p in g["_plies"]:
            m = p["meta"]
            if not m:
                continue
            if m["by"] == "cf" and not m.get("book") and "e" in m and abs(m["e"]) < MATE:
                cf_x.append(m["e"]); cf_y.append(y); cf_ply.append(p["k"])
            if m["by"] == "ut" and "v" in m:
                ut_x.append(m["v"]); ut_y.append(y); ut_ply.append(p["k"])
    a_cf, b_cf = logistic_fit(np.array(cf_x) / 1000, cf_y)
    a_ut, b_ut = logistic_fit(ut_x, ut_y)
    cf_prob = lambda e: 1 / (1 + math.exp(-(a_cf * e / 1000 + b_cf))) if abs(e) < MATE else (1.0 if e > 0 else 0.0)
    ut_prob = lambda v: 1 / (1 + math.exp(-(a_ut * v + b_ut)))
    R["calibration_fit"] = dict(cf=dict(a_per_1000=a_cf, b=b_cf, score_for_75pct=float((math.log(3) - b_cf) / a_cf * 1000)),
                                ut=dict(a=a_ut, b=b_ut))
    # calibration curves: binned mean outcome
    def bins(x, y, edges):
        out = []
        x, y = np.asarray(x), np.asarray(y)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 30:
                out.append(dict(lo=float(lo), hi=float(hi), mean_x=float(x[m].mean()), outcome=float(y[m].mean()), n=int(m.sum())))
        return out
    R["calib_cf"] = bins(cf_x, cf_y, [-99999, -3000, -2000, -1500, -1000, -600, -300, -100, 100, 300, 600, 1000, 1500, 2000, 3000, 99999])
    R["calib_ut"] = bins(ut_x, ut_y, list(np.linspace(-1, 1, 17)))
    # predictive power by phase: Brier score of the fitted probabilities (lower is better)
    phase = []
    for lo, hi in [(0, 15), (15, 25), (25, 35), (35, 45), (45, 55), (55, 82)]:
        cm = [(cf_prob(e), y) for e, y, k in zip(cf_x, cf_y, cf_ply) if lo <= k < hi]
        um = [(ut_prob(v), y) for v, y, k in zip(ut_x, ut_y, ut_ply) if lo <= k < hi]
        if len(cm) > 50 and len(um) > 50:
            phase.append(dict(plies=f"{lo}-{hi - 1}", cf=brier(*zip(*cm)), ut=brier(*zip(*um)),
                              base=brier([np.mean([y for _, y in cm])] * len(cm), [y for _, y in cm]), n_cf=len(cm), n_ut=len(um)))
    R["brier_by_phase"] = phase
    split = {}
    for kind in (True, False):
        rows = []
        for lo, hi in [(0, 20), (20, 30), (30, 40), (40, 82)]:
            c, u = [], []
            for g in games:
                if g["result"]["count_ending"] != kind:
                    continue
                y, n0 = outcome_score(g), len(g["opening"])
                for j, m in enumerate(g["moves"]):
                    if not lo <= n0 + j < hi:
                        continue
                    if m["by"] == "cf" and "e" in m and not m.get("book"):
                        c.append((cf_prob(m["e"]), y))
                    if m["by"] == "ut":
                        u.append((ut_prob(m["v"]), y))
            if len(c) > 50 and len(u) > 50:
                rows.append(dict(plies=f"{lo}-{hi - 1}", cf=brier(*zip(*c)), ut=brier(*zip(*u)), n=len(c) + len(u)))
        split["count" if kind else "line"] = rows
    R["brier_split"] = split

    # ------------------------------------------------------------ disagreement: CF's eval vs uttt.ai's next eval (one ply later)
    dis = []
    for g in games:
        y = outcome_score(g)
        ms = [p for p in g["_plies"] if p["meta"]]
        for a, b in zip(ms, ms[1:]):
            if a["mover"] == "cf" and b["mover"] == "ut" and "e" in a["meta"] and not a["meta"].get("book"):
                pc, pu = cf_prob(a["meta"]["e"]), ut_prob(b["meta"]["v"])
                dis.append((pc, pu, y, a["k"], g["id"]))
    dis = np.array([d[:4] for d in dis])
    gap = dis[:, 0] - dis[:, 1]
    buckets = []
    for lo, hi, label in [(-1, -.3, "uttt.ai far more optimistic for CF"), (-.3, -.1, "uttt.ai more optimistic for CF"),
                          (-.1, .1, "agree (within 10 points)"), (.1, .3, "CF more optimistic for CF"), (.3, 1.01, "CF far more optimistic for CF")]:
        m = (gap >= lo) & (gap < hi)
        if m.sum():
            buckets.append(dict(label=label, n=int(m.sum()), share=float(m.mean()), cf_prob=float(dis[m, 0].mean()),
                                ut_prob=float(dis[m, 1].mean()), outcome=float(dis[m, 2].mean()),
                                cf_brier=brier(dis[m, 0], dis[m, 2]), ut_brier=brier(dis[m, 1], dis[m, 2])))
    R["disagreement"] = buckets
    R["agree_rate"] = float((np.abs(gap) < .1).mean())

    # ------------------------------------------------------------ turning points and recognition lag
    lag = []
    found = []
    for g in games:
        ms = [p for p in g["_plies"] if p["meta"]]
        L = len(g["_plies"])
        if g["result"]["cg"] == 1:  # crossfish won: when did each engine see it?
            cf_sees = next((p["k"] for p in ms if p["mover"] == "cf" and "e" in p["meta"] and cf_prob(p["meta"]["e"]) >= .85), None)
            ut_sees = next((p["k"] for p in ms if p["mover"] == "ut" and ut_prob(p["meta"]["v"]) >= .85), None)
            proof = next((p["k"] for p in ms if p["mover"] == "cf" and p["meta"].get("e", 0) >= MATE), None)
            lag.append(dict(cf=cf_sees, ut=ut_sees, proof=proof, end=L))
            if proof is not None:
                found.append(L - proof)
    have = [d for d in lag if d["cf"] is not None and d["ut"] is not None]
    R["recognition"] = dict(cf_wins=len(lag), both_see=len(have),
                            median_cf=float(np.median([d["cf"] for d in have])) if have else None,
                            median_ut=float(np.median([d["ut"] for d in have])) if have else None,
                            median_lag=float(np.median([d["ut"] - d["cf"] for d in have])) if have else None,
                            ut_never=sum(1 for d in lag if d["ut"] is None),
                            cf_first=sum(1 for d in have if d["cf"] < d["ut"]), ut_first=sum(1 for d in have if d["ut"] < d["cf"]),
                            proof_games=len(found), median_plies_after_proof=float(np.median(found)) if found else None)
    R["recognition_hist"] = Counter(int(np.clip(d["ut"] - d["cf"], -20, 20)) // 2 * 2 for d in have)
    R["recognition_hist"] = sorted([[k, v] for k, v in R["recognition_hist"].items()])

    # ------------------------------------------------------------ crossfish's proofs, and whether uttt.ai converts proven wins
    first_proof, conv, cf_unsound = [], [], []
    for g in games:
        n0, mv = len(g["opening"]), g["moves"]
        kw = next((j for j, m in enumerate(mv) if m["by"] == "cf" and m.get("e", 0) >= MATE), None)
        kl = next((j for j, m in enumerate(mv) if m["by"] == "cf" and m.get("e", 0) <= -MATE), None)
        kd = next((j for j, m in enumerate(mv) if m["by"] == "cf" and m.get("e") == 0 and m.get("d", 0) >= 50), None)
        for kind, k in (("win", kw), ("loss", kl), ("draw", kd)):
            if k is not None:
                first_proof.append(dict(kind=kind, ply=n0 + k, end=n0 + len(mv)))
        if kw is not None and g["result"]["cg"] != 1:
            cf_unsound.append(g["id"])
        if kl is not None:
            later = [m["e"] for m in mv[kl + 1:] if m["by"] == "cf" and "e" in m]
            conv.append(dict(gid=g["id"], ply=n0 + kl + 1, result=g["result"]["cg"], count=g["result"]["count_ending"],
                             retracted=any(e > -MATE for e in later)))
    missed = [c for c in conv if c["result"] != -1]
    R["proofs"] = dict(
        games_win=sum(1 for f in first_proof if f["kind"] == "win"), games_loss=len(conv),
        games_draw=sum(1 for f in first_proof if f["kind"] == "draw"),
        median_ply={k: float(np.median([f["ply"] for f in first_proof if f["kind"] == k])) for k in ("win", "loss", "draw")
                    if any(f["kind"] == k for f in first_proof)},
        median_plies_left=float(np.median([f["end"] - f["ply"] for f in first_proof if f["kind"] == "win"])),
        hist=sorted(Counter(min(60, f["ply"] // 5 * 5) for f in first_proof).items()),
        ut_missed=len(missed), ut_missed_count=sum(c["count"] for c in missed),
        ut_missed_to=dict(Counter("CF won" if c["result"] == 1 else "draw" for c in missed)),
        ut_missed_all_retracted=all(c["retracted"] for c in missed),
        cf_proven_win_not_won=cf_unsound,
        missed_examples=[dict(gid=c["gid"], ply=c["ply"], result=c["result"], count=c["count"]) for c in missed[:10]])

    # ------------------------------------------------------------ uttt.ai blunders: its own value swings toward CF across one CF reply
    swings = []
    for g in games:
        uts = [p for p in g["_plies"] if p["meta"] and p["mover"] == "ut"]
        for a, b in zip(uts, uts[1:]):
            d = ut_prob(b["meta"]["v"]) - ut_prob(a["meta"]["v"])  # rise in CF's chances, per uttt.ai
            swings.append(dict(k=a["k"], d=d, share=a["meta"]["cands"][0][1] if a["meta"].get("cands") else None,
                               gid=g["id"], y=outcome_score(g), sims=a["meta"].get("s")))
    sw = np.array([s["d"] for s in swings])
    big = [s for s in swings if s["d"] >= .25]
    R["ut_swings"] = dict(n=len(swings), big=len(big), big_rate=float(len(big) / max(1, len(swings))),
                          games_with_big=len({s["gid"] for s in big}),
                          big_by_phase=sorted(Counter(min(80, s["k"] // 10 * 10) for s in big).items()),
                          all_by_phase=sorted(Counter(min(80, s["k"] // 10 * 10) for s in swings).items()),
                          confidence_big=float(np.median([s["share"] for s in big if s["share"] is not None])) if big else None,
                          confidence_all=float(np.median([s["share"] for s in swings if s["share"] is not None])))
    # the same for crossfish: its own score swinging toward uttt.ai across one uttt.ai reply
    cf_sw = []
    for g in games:
        cfs = [p for p in g["_plies"] if p["meta"] and p["mover"] == "cf" and "e" in p["meta"] and not p["meta"].get("book")]
        for a, b in zip(cfs, cfs[1:]):
            cf_sw.append(dict(k=a["k"], d=cf_prob(a["meta"]["e"]) - cf_prob(b["meta"]["e"]), gid=g["id"]))
    cbig = [s for s in cf_sw if s["d"] >= .25]
    R["cf_swings"] = dict(n=len(cf_sw), big=len(cbig), big_rate=float(len(cbig) / max(1, len(cf_sw))),
                          games_with_big=len({s["gid"] for s in cbig}),
                          big_by_phase=sorted(Counter(min(80, s["k"] // 10 * 10) for s in cbig).items()),
                          all_by_phase=sorted(Counter(min(80, s["k"] // 10 * 10) for s in cf_sw).items()))

    # ------------------------------------------------------------ does uttt.ai predict crossfish's reply?
    pred = defaultdict(lambda: [0, 0])
    for g in games:
        ps = g["_plies"]
        for a, b in zip(ps, ps[1:]):
            if a["meta"] and a["mover"] == "ut" and len(a["meta"].get("pv", [])) >= 2 and b["meta"] and b["mover"] == "cf":
                bucket = min(60, a["k"] // 10 * 10)
                pred[bucket][0] += a["meta"]["pv"][1] == b["idx"]
                pred[bucket][1] += 1
                pred["all"][0] += a["meta"]["pv"][1] == b["idx"]
                pred["all"][1] += 1
    R["reply_prediction"] = {str(k): v[0] / v[1] for k, v in pred.items() if v[1] >= 30}
    R["reply_prediction_n"] = pred["all"][1]
    # uttt.ai's decisiveness: share of visits on its chosen move
    R["ut_confidence_by_phase"] = {}
    conf = defaultdict(list)
    for g in games:
        for p in g["_plies"]:
            if p["meta"] and p["mover"] == "ut" and p["meta"].get("cands"):
                conf[min(60, p["k"] // 10 * 10)].append(p["meta"]["cands"][0][1])
    R["ut_confidence_by_phase"] = {str(k): float(np.median(v)) for k, v in sorted(conf.items())}

    # ------------------------------------------------------------ style
    style = {}
    for who in ("cf", "ut"):
        ps = [p for g in games for p in g["_plies"] if p["mover"] == who and not p["book"]]
        n = len(ps)
        style[who] = dict(moves=n, captures=sum(p["captured"] for p in ps) / n, gives_free=sum(p["gives_free"] for p in ps) / n,
                          free_choice_rate=sum(p["free_choice"] for p in ps) / n,
                          center_square=sum(p["center"] for p in ps) / n, corner_square=sum(p["corner"] for p in ps) / n)
    R["style"] = style
    # free moves given, per game, vs result
    fm = []
    for g in games:
        cf_free = sum(p["gives_free"] for p in g["_plies"] if p["mover"] == "cf" and not p["book"])
        ut_free = sum(p["gives_free"] for p in g["_plies"] if p["mover"] == "ut" and not p["book"])
        fm.append((ut_free - cf_free, outcome_score(g)))
    fmb = defaultdict(list)
    for d, y in fm:
        fmb[int(np.clip(d, -3, 3))].append(y)
    R["free_moves_vs_result"] = [dict(net=k, games=len(v), cf_score=float(np.mean(v))) for k, v in sorted(fmb.items())]

    # ------------------------------------------------------------ search stats by phase
    sp = defaultdict(lambda: dict(cf_d=[], ut_s=[], cf_ms=[]))
    for g in games:
        for p in g["_plies"]:
            m = p["meta"]
            if not m:
                continue
            b = min(70, p["k"] // 10 * 10)
            if m["by"] == "cf" and not m.get("book") and "d" in m:
                sp[b]["cf_d"].append(m["d"]); sp[b]["cf_ms"].append(m["ms"])
            if m["by"] == "ut" and "s" in m:
                sp[b]["ut_s"].append(m["s"])
    R["search_by_phase"] = [dict(plies=f"{b}-{b + 9}", cf_depth=float(np.median(v["cf_d"])) if v["cf_d"] else None,
                                 cf_ms=float(np.mean(v["cf_ms"])) if v["cf_ms"] else None,
                                 ut_sims=float(np.median(v["ut_s"])) if v["ut_s"] else None, n=len(v["cf_d"]))
                            for b, v in sorted(sp.items())]

    # ------------------------------------------------------------ example positions of strong disagreement (for the viewer)
    ex = []
    for g in games:
        ms = [p for p in g["_plies"] if p["meta"]]
        for a, b in zip(ms, ms[1:]):
            if a["mover"] == "cf" and b["mover"] == "ut" and "e" in a["meta"] and not a["meta"].get("book") and abs(a["meta"]["e"]) < MATE:
                pc, pu = cf_prob(a["meta"]["e"]), ut_prob(b["meta"]["v"])
                ex.append(dict(gid=g["id"], ply=a["k"] + 1, cf_prob=pc, ut_prob=pu, gap=pc - pu, result=g["result"]["cg"],
                               count=g["result"]["count_ending"]))
    ex.sort(key=lambda e: -abs(e["gap"]))
    seen, top = set(), []
    for e in ex:
        if e["gid"] not in seen:
            seen.add(e["gid"]); top.append(e)
        if len(top) == 12:
            break
    R["examples"] = top

    json.dump(R, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: R[k] for k in ("proofs", "brier_split", "endings", "by_color", "length", "pentanomial", "calibration_fit", "brier_by_phase",
                                         "disagreement", "agree_rate", "recognition", "ut_swings", "cf_swings",
                                         "reply_prediction", "ut_confidence_by_phase", "style", "free_moves_vs_result",
                                         "search_by_phase", "count_boards")}, indent=1, default=float))


if __name__ == "__main__":
    main()
