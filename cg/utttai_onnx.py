"""uttt.ai's Neural MCTS (utttpy/selfplay/neural_monte_carlo_tree_search.py,
Apache-2.0) with the policy-value network run through onnxruntime instead of
PyTorch. Selection (Q + U with the max(0.01, p) floor), backup, tree reuse via
synchronize() and argmax move choice follow the original line for line; only
_evaluate changes, to call the deployed policy_value_net_stage2.onnx.

solver=True (or UTTTAI_SOLVER=1) adds an MCTS-solver: every node may carry an
exact result for its side to move (+1 win, -1 loss, 0 draw). Game-ending
children are scored when created, a node with a child lost for the opponent is
a proven win, a node whose children are all proven takes the best of them, and
proven nodes are backed up with their exact value instead of being searched.
Selection never enters a child proven won for the opponent while any other
child remains, and best_action() plays a proven win and avoids proven losses.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
# The game module is this repository's utttpy (CodinGame rules: a full board goes to the subgame count).
sys.path.insert(0, str(HERE.parent))
from utttpy.game.action import Action  # noqa: E402
from utttpy.game.helpers import get_state_ndarray_4x9x9, row_index, col_index  # noqa: E402
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe  # noqa: E402


def make_session(path: Path = None) -> ort.InferenceSession:
    # UTTTAI_NET overrides the model (e.g. a fine-tuned train/net1.onnx).
    path = path or HERE / os.environ.get("UTTTAI_NET", "policy_value_net_stage2.onnx")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # one core per game worker, like the site's single-threaded search
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class Node:
    __slots__ = ("uttt", "action", "action_probability", "child_nodes", "visit_count",
                 "state_value", "state_value_sum", "state_value_mean", "proven")

    def __init__(self, uttt: UltimateTicTacToe, action: Optional[Action] = None):
        self.uttt = uttt
        self.action = action
        self.action_probability = None
        self.child_nodes: List[Node] = []
        self.visit_count = 0
        self.state_value = None
        self.state_value_sum = 0.0
        self.state_value_mean = 0.0
        self.proven: Optional[float] = None  # exact result for the side to move, if solved

    def is_leaf(self) -> bool:
        return len(self.child_nodes) == 0

    def expand(self, solver: bool = False) -> None:
        if not self.is_leaf() or self.uttt.is_terminated():
            return
        for legal_action in self.uttt.get_legal_actions():
            uttt = self.uttt.clone()
            uttt.execute(action=legal_action, verify=False)
            child = Node(uttt=uttt, action=legal_action)
            if solver and uttt.is_terminated():
                child.proven = terminal_value(uttt)
            self.child_nodes.append(child)


def terminal_value(uttt: UltimateTicTacToe) -> float:
    """Result of a finished game for the side to move. Under uttt.ai's rules the
    last mover always wins; under CodinGame rules a count ending can go to the
    side to move."""
    if uttt.is_result_draw():
        return 0.0
    return 1.0 if uttt.result == uttt.next_symbol else -1.0


class NMCTS:
    def __init__(self, uttt: UltimateTicTacToe, num_simulations: int, session: ort.InferenceSession,
                 exploration_strength: float = 2.0, solver: Optional[bool] = None):
        self.solver = os.environ.get("UTTTAI_SOLVER") == "1" if solver is None else solver
        self.root = Node(uttt=uttt.clone())
        self.num_simulations = num_simulations
        self.exploration_strength = exploration_strength
        self.session = session

    def run(self) -> None:
        for _ in range(self.num_simulations - self.root.visit_count):
            if self.solved():
                break
            self._simulate()

    def solved(self) -> bool:
        """The root's result is proven, so more search cannot change the move."""
        return self.solver and self.root.proven is not None and bool(self.root.child_nodes)

    def best_action(self) -> Action:
        kids = self.root.child_nodes
        if self.solver:
            wins = [c for c in kids if c.proven == -1.0]  # the opponent is lost after these
            if wins:
                kids = wins
            else:
                kids = [c for c in kids if c.proven != 1.0] or kids  # avoid proven losses
        top = max(child.visit_count for child in kids)
        best = random.choice([c for c in kids if c.visit_count >= top])
        return best.action

    def synchronize(self, uttt: UltimateTicTacToe) -> None:
        for child in self.root.child_nodes:
            if uttt.is_equal_to(child.uttt):
                self.root = child
                return
        self.root = Node(uttt=uttt.clone())

    def _simulate(self) -> None:
        if self.solver:
            return self._simulate_solver()
        path = []
        node = self.root
        while not node.is_leaf():
            path.append(node)
            sqrt_parent = math.sqrt(node.visit_count)
            scores = [
                -c.state_value_mean
                + self.exploration_strength * max(0.01, c.action_probability) * sqrt_parent / (c.visit_count + 1)
                for c in node.child_nodes
            ]
            top = max(scores)
            node = node.child_nodes[random.choice([i for i, s in enumerate(scores) if s >= top])]
        path.append(node)
        node.expand()
        self._evaluate(node)
        sign = 1
        for n in reversed(path):
            n.visit_count += 1
            n.state_value_sum += sign * node.state_value
            n.state_value_mean = n.state_value_sum / n.visit_count
            sign = -sign

    def _simulate_solver(self) -> None:
        path = []
        node = self.root
        while not node.is_leaf() and node.proven is None:
            path.append(node)
            sqrt_parent = math.sqrt(node.visit_count)
            best, best_s = [], -math.inf
            for i, c in enumerate(node.child_nodes):
                if c.proven == 1.0:
                    continue  # won for the opponent: never worth entering
                q = -c.proven if c.proven is not None else -c.state_value_mean
                s = q + self.exploration_strength * max(0.01, c.action_probability) * sqrt_parent / (c.visit_count + 1)
                if s > best_s:
                    best, best_s = [i], s
                elif s == best_s:
                    best.append(i)
            node = node.child_nodes[random.choice(best)]
        path.append(node)
        if node.proven is not None:
            value = node.proven  # solved: back up the exact result, no search needed
        else:
            node.expand(solver=True)
            if node.uttt.is_terminated():
                node.proven = node.state_value = value = terminal_value(node.uttt)
            elif any(c.proven == -1.0 for c in node.child_nodes):
                node.proven = node.state_value = value = 1.0  # a move ends the game in our favour
            else:
                self._evaluate(node)
                value = node.state_value
        sign = 1
        for n in reversed(path):
            n.visit_count += 1
            n.state_value_sum += sign * value
            n.state_value_mean = n.state_value_sum / n.visit_count
            sign = -sign
        # Propagate proofs upward; stop at the first node that stays unproven.
        for n in reversed(path):
            if n.proven is None and n.child_nodes:
                if any(c.proven == -1.0 for c in n.child_nodes):
                    n.proven = 1.0
                elif all(c.proven is not None for c in n.child_nodes):
                    n.proven = -min(c.proven for c in n.child_nodes)
            if n.proven is None:
                break

    def _evaluate(self, node: Node) -> None:
        if node.uttt.is_terminated():
            node.state_value = terminal_value(node.uttt)
            return
        x = get_state_ndarray_4x9x9(uttt=node.uttt).astype(np.float32)[None]
        logits, value = self.session.run(["policy_logits", "state_value"], {"input": x})
        picked = np.array([logits[0, row_index(c.action.index), col_index(c.action.index)]
                           for c in node.child_nodes], dtype=np.float64)
        probs = np.exp(picked - picked.max())
        probs /= probs.sum()
        for c, p in zip(node.child_nodes, probs):
            c.action_probability = float(p)
        node.state_value = float(value[0])


if __name__ == "__main__":
    import time
    sess = make_session()
    uttt = UltimateTicTacToe()
    for sims in (1, 100, 1000):
        search = NMCTS(uttt, sims, sess)
        t0 = time.perf_counter()
        search.run()
        a = search.best_action()
        print(f"{sims:5d} simulations from the empty board: {time.perf_counter() - t0:6.2f} s, "
              f"plays index {a.index} (miniboard {a.index // 9}, square {a.index % 9})")
