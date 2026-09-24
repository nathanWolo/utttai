"""crossfish (CodinGame bot, match protocol) vs uttt.ai (NMCTS + stage-2 ONNX net).

Games start from positions in crossfish's balanced SPRT opening book; each
opening is played twice with colors swapped. uttt.ai's rules module referees.
Every game is scored under both rule sets: they end at the same point and only
disagree on a full board with no three-in-a-row (uttt.ai: draw; CodinGame:
whoever won more miniboards).

Live feed for the viewer (serve_viewer.py): after every move each game is written
to <log>.live/g<id>.json, and the running tally to <log>.live/summary.json.

<utttai_sims> is a simulation count, or a time budget such as 90ms: uttt.ai then
simulates until the budget is spent (checked after every simulation), so both
engines get the same wall time per move on one core each.

usage: python bench_vs_utttai.py <utttai_sims | NNNms> <crossfish_ms> <openings> <workers> <log> [crossfish_exe]
"""
import json
import math
import os
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = Path(r"C:\Users\natha\crossfish\cpp_impl\opening_book.bin")
CROSSFISH = HERE / "crossfish_cg.exe"


def load_openings(n):
    data = BOOK.read_bytes()
    count = int.from_bytes(data[8:12], "little")
    step = count // n
    out = []
    for i in range(n):
        rec = data[32 + 16 * (i * step): 32 + 16 * (i * step + 1)]
        out.append(list(rec[1:1 + rec[0]]))
    return out


def elo(w, d, l):
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s * s) / n
    sd = math.sqrt(var / n)
    f = lambda p: -400 * math.log10(1 / min(0.999, max(0.001, p)) - 1)
    return f(s), (f(s + 1.96 * sd) - f(s - 1.96 * sd)) / 2


def write_json(path, obj):
    """Atomic replace, retried: a reader holding the file open blocks os.replace on Windows."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    tmp.unlink(missing_ok=True)


def parse_budget(arg):
    """'1000' -> (1000 simulations, None); '90ms' -> (unbounded, 90 ms)."""
    return (10 ** 9, int(arg[:-2])) if str(arg).endswith("ms") else (int(arg), None)


def worker(jobs, results, sims, cf_ms, exe, live=None, wid=0):
    sims, ut_ms = parse_budget(sims)
    sys.path.insert(0, str(HERE))
    from utttai_onnx import NMCTS, make_session, UltimateTicTacToe
    from utttpy.game.action import Action
    sess = make_session()
    proc = subprocess.Popen([exe, "match"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(line):
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    while True:
        job = jobs.get()
        if job is None:
            break
        game_id, opening, cf_is_x = job
        random.seed(game_id)
        u = UltimateTicTacToe()
        send("NEW")
        assert proc.stdout.readline().strip() == "READY"
        for idx in opening:
            u.execute(Action(symbol=u.next_symbol, index=idx))
            send(f"APPLY {idx // 9} {idx % 9}")
        search = NMCTS(u, sims, sess)
        cf_ms_used, ut_s_used, cf_moves, ut_moves = 0.0, 0.0, 0, 0
        feed = dict(id=game_id, worker=wid, cf_is_x=cf_is_x, opening=opening, moves=[], status="playing",
                    thinking="crossfish" if u.is_next_symbol_X() == cf_is_x else "utttai",
                    started=time.time(), updated=time.time())
        if live:
            write_json(live / f"g{game_id:04d}.json", feed)
        while not u.is_terminated():
            cf_turn = u.is_next_symbol_X() == cf_is_x
            t0 = time.perf_counter()
            if cf_turn:
                send(f"GO {cf_ms}")
                reply = proc.stdout.readline().split()
                mb, sq = int(reply[0]), int(reply[1])
                idx = mb * 9 + sq
                cf_ms_used += (time.perf_counter() - t0) * 1000
                cf_moves += 1
                move = dict(i=idx, by="cf", ms=round((time.perf_counter() - t0) * 1000))
                if len(reply) >= 6:  # crossfish_cg_meta.exe: depth, root score (crossfish's view), nodes, book flag
                    move.update(d=int(reply[2]), e=int(reply[3]), n=int(reply[4]), book=reply[5] == "1")
            else:
                search.synchronize(u)
                before = search.root.visit_count
                if ut_ms is None:
                    search.run()
                else:
                    deadline = t0 + ut_ms / 1000
                    while (search.root.visit_count - before < 2 or time.perf_counter() < deadline) and not search.solved():
                        search._simulate()
                idx = search.best_action().index
                ut_s_used += time.perf_counter() - t0
                ut_moves += 1
                root = search.root
                kids = sorted(root.child_nodes, key=lambda c: -c.visit_count)[:4]
                # root value is for uttt.ai (the side to move); the feed stores crossfish's view
                pv, node = [], root
                while node.child_nodes and len(pv) < 24:
                    node = max(node.child_nodes, key=lambda c: c.visit_count)
                    if node.visit_count == 0:
                        break
                    pv.append(node.action.index)
                move = dict(i=idx, by="ut", ms=round((time.perf_counter() - t0) * 1000),
                            v=round(-root.state_value_mean, 4), n=root.visit_count, s=root.visit_count - before, pv=pv,
                            cands=[[c.action.index, round(c.visit_count / max(1, root.visit_count - 1), 3)] for c in kids])
                send(f"APPLY {idx // 9} {idx % 9}")
            if idx not in u.get_legal_indexes():
                raise RuntimeError(f"illegal move {idx} by {'crossfish' if cf_turn else 'utttai'}")
            u.execute(Action(symbol=u.next_symbol, index=idx), verify=False)
            search.synchronize(u)
            if live:
                feed["moves"].append(move)
                feed["thinking"] = "utttai" if cf_turn else "crossfish"
                feed["updated"] = time.time()
                if not u.is_terminated():
                    write_json(live / f"g{game_id:04d}.json", feed)
        # +1 crossfish win, 0 draw, -1 loss, scored from the final supergame so
        # it does not depend on which rules module (original or fork) refereed.
        sup = list(u.state[81:90])
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        line_winner = next((sup[a] for a, b, c in lines if sup[a] in (1, 2) and sup[a] == sup[b] == sup[c]), None)
        cf_sym = 1 if cf_is_x else 2
        if line_winner is not None:
            ut_score = cg_score = 1 if line_winner == cf_sym else -1
            count_ending = False
        else:
            ut_score = 0
            mine, theirs = sup.count(cf_sym), sup.count(3 - cf_sym)
            cg_score = (mine > theirs) - (mine < theirs)
            count_ending = True
        if live:
            feed.update(status="done", thinking=None, updated=time.time(),
                        result=dict(cg=cg_score, ut=ut_score, count_ending=count_ending))
            write_json(live / f"g{game_id:04d}.json", feed)
        results.put((game_id, cf_is_x, ut_score, cg_score, count_ending,
                     cf_ms_used / max(1, cf_moves), ut_s_used / max(1, ut_moves), len(opening)))
    proc.kill()


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def wait_for_watch_match(log_path):
    """A timed benchmark must not share the CPU with a viewing match (watch_*.log):
    crossfish would search less in its fixed time. Wait while one is running."""
    busy = HERE / "watch_busy.pid"
    if Path(log_path).name.startswith("watch") or not busy.exists():
        return
    pid = int(busy.read_text().strip() or 0)
    while pid and pid_alive(pid):
        print(f"[{time.strftime('%H:%M:%S')}] waiting for the watch match (pid {pid}) to finish", flush=True)
        time.sleep(30)
    busy.unlink(missing_ok=True)


def main():
    sims, cf_ms, n_open, workers, log_path = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                              int(sys.argv[4]), sys.argv[5])
    wait_for_watch_match(log_path)
    exe = str(HERE / (sys.argv[6] if len(sys.argv) > 6 else "crossfish_cg.exe"))
    openings = load_openings(n_open)
    jobs, results = mp.Queue(), mp.Queue()
    total = 2 * n_open
    for i, op in enumerate(openings):
        jobs.put((2 * i, op, True))
        jobs.put((2 * i + 1, op, False))
    for _ in range(workers):
        jobs.put(None)
    live = Path(log_path).with_suffix(".live")
    live.mkdir(parents=True, exist_ok=True)
    for f in live.glob("*.json"):
        f.unlink()
    net = os.environ.get("UTTTAI_NET", "policy_value_net_stage2.onnx")
    summary = dict(title=f"{Path(exe).stem} vs uttt.ai {Path(net).stem}", crossfish=Path(exe).stem,
                   utttai=Path(net).stem, cf_ms=cf_ms, sims=sims, ut_ms=parse_budget(sims)[1], total=2 * n_open, workers=workers,
                   started=time.time(), done=0, cg=[0, 0, 0], ut=[0, 0, 0], elo_cg=None, elo_ut=None,
                   eta=None, finished=False)
    write_json(live / "summary.json", summary)
    procs = [mp.Process(target=worker, args=(jobs, results, sims, cf_ms, exe, live, w)) for w in range(workers)]
    for p in procs:
        p.start()
    log = open(log_path, "w", encoding="utf-8")

    def out(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out(f"{Path(exe).name} vs uttt.ai (CodinGame rules), net {__import__('os').environ.get('UTTTAI_NET', 'policy_value_net_stage2.onnx')}: crossfish {cf_ms} ms/move vs uttt.ai {sims if str(sims).endswith("ms") else str(sims) + " simulations"}/move; {n_open} openings x 2 colors = {total} games, {workers} workers")
    t0 = time.time()
    rows = []
    for k in range(1, total + 1):
        rows.append(results.get())
        wdl3 = lambda xs: [xs.count(1), xs.count(0), xs.count(-1)]
        summary.update(done=k, cg=wdl3([r[3] for r in rows]), ut=wdl3([r[2] for r in rows]),
                       elo_cg=list(elo(*wdl3([r[3] for r in rows]))), elo_ut=list(elo(*wdl3([r[2] for r in rows]))),
                       eta=t0 + (time.time() - t0) / k * total, finished=k == total)
        write_json(live / "summary.json", summary)
        if k % 10 == 0 or k == total:
            ut = [r[2] for r in rows]
            cg = [r[3] for r in rows]
            wdl = lambda xs: (xs.count(1), xs.count(0), xs.count(-1))
            e_ut, ci_ut = elo(*wdl(ut))
            e_cg, ci_cg = elo(*wdl(cg))
            eta = time.strftime("%H:%M", time.localtime(t0 + (time.time() - t0) / k * total))
            out(f"[{time.strftime('%H:%M:%S')}] {k}/{total} games  ETA {eta}  "
                f"crossfish W/D/L uttt.ai rules {wdl(ut)} Elo {e_ut:+.0f} +/- {ci_ut:.0f} | "
                f"CodinGame rules {wdl(cg)} Elo {e_cg:+.0f} +/- {ci_cg:.0f}")
    for p in procs:
        p.join()
    ut = [r[2] for r in rows]
    cg = [r[3] for r in rows]
    wdl = lambda xs: (xs.count(1), xs.count(0), xs.count(-1))
    count_endings = sum(r[4] for r in rows)
    as_x = [r for r in rows if r[1]]
    as_o = [r for r in rows if not r[1]]
    out("")
    out(f"uttt.ai rules  : crossfish W/D/L {wdl(ut)}  Elo {elo(*wdl(ut))[0]:+.1f} +/- {elo(*wdl(ut))[1]:.1f}")
    out(f"CodinGame rules: crossfish W/D/L {wdl(cg)}  Elo {elo(*wdl(cg))[0]:+.1f} +/- {elo(*wdl(cg))[1]:.1f}")
    out(f"full board, no line (the only rule difference): {count_endings}/{total} games")
    out(f"crossfish as X (moves first from the opening position's side): {wdl([r[2] for r in as_x])} uttt.ai rules")
    out(f"crossfish as O: {wdl([r[2] for r in as_o])} uttt.ai rules")
    out(f"mean think time per move: crossfish {sum(r[5] for r in rows) / total:.1f} ms, "
        f"uttt.ai {sum(r[6] for r in rows) / total:.3f} s")


if __name__ == "__main__":
    main()
