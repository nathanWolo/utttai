"""Generate and build the crossfish variants the match tools use, from crossfish's
CodinGame bot (cpp_impl/codingame_nnue.cpp). The engine itself is untouched.

  codingame_nnue.cpp -> crossfish_cg.exe  the plain CodinGame bot, unmodified
  cf_meta.cpp  -> crossfish_cg_meta.exe   match mode also reports
                                          "<mb> <sq> <depth> <root score> <nodes> <book 0/1>"
  cf_debug.cpp -> crossfish_cg_debug.exe  cf_meta plus debugging aids:
      CF_TRACE=1        per-iteration trace on stderr (window, result, fail-low/high)
      THINK <ms>        search like GO and report, without playing the move
      PV                print the transposition-table line from the current position
      CF_NO_PRUNE=1     disable eval-based pruning (g_disable_eval_prune)
      CF_NO_FGW=1       disable the forced-global-win-after-reply shortcut
      CF_NO_IGW=1       disable the immediate-global-win shortcut
      CF_NO_TTMATE=1    ignore transposition-table cutoffs on mate scores

usage: python make_crossfish_variants.py [crossfish_repo=C:/Users/natha/crossfish] [--no-build]
Writes the .cpp files and executables into cg/ (the parent of this tools/ folder).
"""
import subprocess
import sys
from pathlib import Path

CG = Path(__file__).resolve().parent.parent
REPO = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "C:/Users/natha/crossfish"))
TOOLCHAIN = REPO / "toolchains/llvm-mingw-20260616-ucrt-x86_64/bin"
FLAGS = ["-O3", "-std=c++17", "-mavx2", "-mbmi", "-mbmi2", "-mlzcnt", "-mpopcnt", "-pthread",
         "-Wno-unknown-pragmas", "-Wno-ignored-attributes", "-Wl,--stack,16777216"]


def rep(s, a, b):
    assert s.count(a) == 1, f"patch anchor not found exactly once (crossfish source changed?):\n{a}"
    return s.replace(a, b)


META = [(
    '''            Move best = engine.getMove(board, std::chrono::milliseconds(ms));
            Move book_move;
            if (pb_lookup(board, book_move)) best = book_move;
            board.makeMove(best);
            std::cout << (int)best.mini_board << " " << (int)best.square << std::endl;''',
    '''            Move best = engine.getMove(board, std::chrono::milliseconds(ms));
            Move book_move;
            bool from_book = pb_lookup(board, book_move);
            if (from_book) best = book_move;
            board.makeMove(best);
            // viewer metadata: search depth, root score (side to move), nodes, book flag
            std::cout << (int)best.mini_board << " " << (int)best.square << " " << engine.depth << " "
                      << engine.root_score << " " << engine.nodes << " " << (from_book ? 1 : 0) << std::endl;''')]

DEBUG = [
    ('''            while (!time_up() && (depth < 50)) {
                int eval = search(board, depth, 0, alpha, beta);
                if (stopped) break;''',
     '''            while (!time_up() && (depth < 50)) {
                int eval = search(board, depth, 0, alpha, beta);
                if (getenv("CF_TRACE")) {
                    auto el = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
                    std::cerr << "iter depth " << depth << " window [" << alpha << ", " << beta << "] -> " << eval
                              << (stopped ? " (stopped)" : eval <= alpha ? " FAIL-LOW" : eval >= beta ? " FAIL-HIGH" : "")
                              << " root_score " << root_score << " best " << (int)root_best_move.mini_board << "." << (int)root_best_move.square
                              << " nodes " << nodes << " t " << el / 1000.0 << "ms" << std::endl;
                }
                if (stopped) break;'''),
    ('''        } else if (cmd == "GO") {''',
     '''        } else if (cmd == "PV") {
            engine.dump_pv(board, 40);
            std::cout << "PVEND" << std::endl;
        } else if (cmd == "THINK") {  // search the position like GO, report, but do not play the move
            int ms;
            std::cin >> ms;
            Move best = engine.getMove(board, std::chrono::milliseconds(ms));
            std::cout << (int)best.mini_board << " " << (int)best.square << " " << engine.depth << " "
                      << engine.root_score << " " << engine.nodes << std::endl;
        } else if (cmd == "GO") {'''),
    ('''        CrossfishDev() {
            d16_mini_load_packed();''',
     '''        void dump_pv(GlobalBoard b, int maxlen) {
            for (int k = 0; k < maxlen; k++) {
                FastBoard fb(b);
                CompactTTBucket &bk = transposition_table[fb.tt_hash & (tt_bucket_count - 1)];
                int hit = -1;
                for (int e = 0; e < 2; e++) if (bk.entries[e].zobrist_hash == fb.tt_hash && fb.tt_hash) hit = e;
                if (hit < 0) { std::cout << "PV " << k << " miss" << std::endl; return; }
                CompactTTEntry en = bk.entries[hit];
                Move mv = unpack_tt_move(en.best_move);
                std::cout << "PV " << k << " move " << (int)mv.mini_board << " " << (int)mv.square << " score " << en.score
                          << " depth " << en.depth << " flag " << (int)en.flag << std::endl;
                if (mv.mini_board > 8) return;
                b.makeMove(mv);
            }
        }

        CrossfishDev() {
            d16_mini_load_packed();'''),
    ('''static bool g_disable_eval_prune = false;''',
     '''static bool g_disable_eval_prune = getenv("CF_NO_PRUNE") != nullptr;
static bool g_no_fgw = getenv("CF_NO_FGW") != nullptr;
static bool g_no_igw = getenv("CF_NO_IGW") != nullptr;
static bool g_no_ttmate = getenv("CF_NO_TTMATE") != nullptr;'''),
    ('''                if (opponent_global_targets
                    && has_immediate_global_win(board, opponent_global_targets)) {''',
     '''                if (!g_no_igw && opponent_global_targets
                    && has_immediate_global_win(board, opponent_global_targets)) {'''),
    ('''                else if (has_forced_global_win_after_reply(board, stm)) {''',
     '''                else if (!g_no_fgw && has_forced_global_win_after_reply(board, stm)) {'''),
    ('''            if (tt_hit && (entry.depth >= depth)) {
                // Flags match''',
     '''            if (tt_hit && (entry.depth >= depth) && !(g_no_ttmate && abs(entry.score) >= 90000)) {
                // Flags match'''),
]


def main():
    src = (REPO / "cpp_impl/codingame_nnue.cpp").read_text(encoding="utf-8")
    meta = src
    for a, b in META:
        meta = rep(meta, a, b)
    debug = meta
    for a, b in DEBUG:
        debug = rep(debug, a, b)
    (CG / "cf_meta.cpp").write_text(meta, encoding="utf-8")
    (CG / "cf_debug.cpp").write_text(debug, encoding="utf-8")
    print("wrote cf_meta.cpp and cf_debug.cpp")
    if "--no-build" in sys.argv:
        return
    import os
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ["PATH"])
    for cpp, exe in ((REPO / "cpp_impl/codingame_nnue.cpp", "crossfish_cg.exe"),
                     (CG / "cf_meta.cpp", "crossfish_cg_meta.exe"), (CG / "cf_debug.cpp", "crossfish_cg_debug.exe")):
        subprocess.run(["clang++", *FLAGS, f"-I{REPO / 'cpp_impl'}", "-o", str(CG / exe), str(cpp)], check=True, env=env)
        print("built", exe)


if __name__ == "__main__":
    main()
