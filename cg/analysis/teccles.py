"""Does deep search agree with the "Teccles" opening heuristic?

Teccles: early in the game, answer being sent to miniboard b by playing square
b of it, which sends the opponent straight back to board b. The move exists
when the mover is forced into board b and that board's square b is empty.

Positions: the 300 opening-book positions (plies 4-10) of early_game.py, whose
90 ms moves and deep references are reused, plus <new> positions at plies 1-3
sampled from uttt.ai's policy from the empty board. For every position with a
Teccles move:

  - pick rates: crossfish 90 ms / 5 s and uttt.ai 90 ms / 5,000 simulations,
    against chance (1 / number of legal moves)
  - uttt.ai 5,000 simulations: the Teccles move's rank and share of visits,
    and its value against the best move's (mover's view)
  - crossfish: a 2 s search after the Teccles move against the same search
    after crossfish's own 5 s move (mover's view)

usage: python teccles.py <crossfish_debug.exe> <early_game_300.json> <out.json> [new=200] [workers=8]
env: UTTTAI_NET, PATH (toolchain DLLs)
"""
import json
import multiprocessing as mp
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import early_game as EG  # noqa: E402


def game(moves):
    import utttai_onnx as U
    u = U.UltimateTicTacToe()
    for i in moves:
        u.execute(U.Action(symbol=u.next_symbol, index=i), verify=False)
    return u


def teccles_move(moves):
    """Index of the Teccles move, or None (free choice, or the square is taken)."""
    if not moves:
        return None
    u = game(moves)
    legal = u.get_legal_indexes()
    boards = {i // 9 for i in legal}
    if len(boards) != 1:
        return None
    b = boards.pop()
    return b * 9 + b if b * 9 + b in legal else None


def sample_early(n, seed=7):
    """Lines of 1-3 plies drawn from uttt.ai's policy (one network call per ply)."""
    import utttai_onnx as U
    rng = random.Random(seed)
    sess = U.make_session()
    out = []
    while len(out) < n:
        k = rng.choice((1, 2, 3))
        moves = []
        for _ in range(k):
            s = U.NMCTS(game(moves), 1, sess)
            s.run()
            kids = s.root.child_nodes
            moves.append(rng.choices([c.action.index for c in kids], weights=[c.action_probability for c in kids])[0])
        if teccles_move(moves) is not None:
            out.append(moves)
    return out


def deep(args):
    moves, tec = args
    import utttai_onnx as U
    u = game(moves)
    s = U.NMCTS(u, 5000, EG.SESS)
    s.run()
    kids = sorted(s.root.child_nodes, key=lambda c: -c.visit_count)
    total = sum(c.visit_count for c in kids)
    q = lambda c: -c.state_value_mean  # mover's view
    tc = next(c for c in kids if c.action.index == tec)
    best = kids[0]
    cf5 = EG.cf_think(moves, 5000)[0]
    cf_after = lambda mv: -EG.cf_think(moves + [mv], 2000)[1]
    return dict(legal=len(kids), ut_rank=kids.index(tc) + 1, ut_share=tc.visit_count / max(1, total),
                ut_q_tec=q(tc), ut_q_best=q(best), ut5k=best.action.index, cf5s=cf5,
                cf_tec=cf_after(tec), cf_best=cf_after(cf5) if cf5 != tec else None)


def main():
    cf, prev_json, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_new = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    EG.init_worker(cf)
    rows = [dict(moves=r["moves"], ply=r["ply"], cf90=r["cf90"], ut90=r["ut90"]) for r in json.load(open(prev_json))]
    rows = [r for r in rows if teccles_move(r["moves"]) is not None]
    new = sample_early(n_new)
    for m in new:  # the fast moves for the new positions, one search at a time
        rows.append(dict(moves=m, ply=len(m), cf90=EG.cf_think(m, 90)[0], ut90=EG.ut_search(m, ms=90)[0]))
    for r in rows:
        r["tec"] = teccles_move(r["moves"])
    print(f"{len(rows)} positions with a Teccles move ({len(new)} new at plies 1-3)", flush=True)
    with mp.Pool(workers, initializer=EG.init_worker, initargs=(cf,)) as pool:
        for k, res in enumerate(pool.imap(deep, [(r["moves"], r["tec"]) for r in rows])):
            rows[k].update(res)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(rows)}", flush=True)
    json.dump(rows, open(out_path, "w"))
    report(rows)


def report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups["plies 1-3" if r["ply"] <= 3 else "plies 4-6" if r["ply"] <= 6 else "plies 7-10"].append(r)
    groups["all"] = rows
    print(f"\nTeccles move available in {len(rows)} positions")
    hdr = "picks the Teccles move:"
    print(f"{hdr:26s} {'chance':>7s} {'cf 90ms':>8s} {'cf 5s':>7s} {'ut 90ms':>8s} {'ut 5k':>7s}")
    for name in ("plies 1-3", "plies 4-6", "plies 7-10", "all"):
        g = groups.get(name, [])
        if not g:
            continue
        rate = lambda key: 100 * sum(r[key] == r["tec"] for r in g) / len(g)
        chance = 100 * statistics.mean(1 / r["legal"] for r in g)
        print(f"  {name + f' ({len(g)})':24s} {chance:6.1f}% {rate('cf90'):7.1f}% {rate('cf5s'):6.1f}% "
              f"{rate('ut90'):7.1f}% {rate('ut5k'):6.1f}%")
    print("\nuttt.ai 5,000 sims on the Teccles move:")
    for name in ("plies 1-3", "plies 4-6", "plies 7-10", "all"):
        g = groups.get(name, [])
        if not g:
            continue
        gap = [r["ut_q_best"] - r["ut_q_tec"] for r in g]
        print(f"  {name:11s} median rank {statistics.median(r['ut_rank'] for r in g):3.0f} of "
              f"{statistics.median(r['legal'] for r in g):2.0f}; median visit share {100 * statistics.median(r['ut_share'] for r in g):5.1f}%; "
              f"value gap to best: median {statistics.median(gap):.3f}, within 0.02 in "
              f"{100 * sum(x <= 0.02 for x in gap) / len(g):4.1f}%, over 0.10 in {100 * sum(x > 0.10 for x in gap) / len(g):4.1f}%")
    print("\ncrossfish 2 s after the Teccles move vs after its own 5 s move (score gap, mover's view):")
    for name in ("plies 1-3", "plies 4-6", "plies 7-10", "all"):
        g = [r for r in groups.get(name, []) if r["cf_best"] is not None]
        if not g:
            continue
        gap = [r["cf_best"] - r["cf_tec"] for r in g]
        print(f"  {name:11s} ({len(g)} where crossfish 5 s prefers another move): median gap {statistics.median(gap):5.0f}; "
              f"Teccles within 100 in {100 * sum(x <= 100 for x in gap) / len(g):4.1f}%, worse by over 500 in "
              f"{100 * sum(x > 500 for x in gap) / len(g):4.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.load(open(sys.argv[2])))
    else:
        main()
