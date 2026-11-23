"""Restricted Nash Response on Kuhn poker: gain against a biased opponent vs exploitability.

Player 0 trains against a player 1 who plays a fixed biased strategy with probability p
and adapts with probability 1-p. p=0 gives Nash, p=1 the best response to the opponent.
Compared with the naive mix (1-lambda) Nash + lambda best response.

    python experiments/rnr.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games.kuhn import KuhnGame
from training.cfr_training import RestrictedNashTrainer
from training.evaluation import best_response_value, expected_value

FIGURES = ROOT / "figures"
P_VALUES = np.linspace(0, 1, 11)
N_ITER = 2_000


def from_bet_probs(bet: dict) -> dict[str, np.ndarray]:
    """Strategy {key: [P(b), P(p)]} from the probability of "b" in each infoset."""
    return {k: np.array([b, 1 - b]) for k, b in bet.items()}


def opponents() -> dict[str, dict[str, np.ndarray]]:
    """Player 1 at equilibrium, with one or several biases."""
    return {
        "folds too much (king facing a bet)": from_bet_probs(
            {"0p": 1/3, "1p": 0, "2p": 1, "0b": 0, "1b": 0, "2b": 1}),
        "bluffs too much (queen after a check)": from_bet_probs(
            {"0p": 1, "1p": 0, "2p": 1, "0b": 0, "1b": 1/3, "2b": 1}),
        "calls too much, never bluffs, slow-plays the ace": from_bet_probs(
            {"0p": 0, "1p": 0, "2p": 1/2, "0b": 0, "1b": 1, "2b": 1}),
    }


def measure(game, profile, fixed) -> tuple[float, float]:
    """(exploitability of player 0, gain of player 0 against the fixed opponent)"""
    expl = best_response_value(game, profile, 1) - 1/18
    gain = expected_value(game, {**profile, **fixed})
    return expl, gain


def own_reach(game, profile, key) -> float:
    """Probability that the player of this key plays the actions leading to it."""
    card, history = key[0], key[1:]
    player = game.current_player(history)
    reach = 1
    for i, action in enumerate(history):
        if game.current_player(history[:i]) == player:
            reach = reach * profile[card + history[:i]][game.ACTIONS.index(action)]
    return reach


def mix(game, s_nash, s_br, lam) -> dict[str, np.ndarray]:
    """Strategy that plays s_br with probability lam and s_nash otherwise (drawn at the start
    of the hand), written infoset by infoset: each one is weighted by the probability of reaching it."""
    res = {}
    for key in s_nash:
        w_nash = (1 - lam) * own_reach(game, s_nash, key)
        w_br = lam * own_reach(game, s_br, key)
        if w_nash + w_br == 0:
            res[key] = s_nash[key]
        else:
            res[key] = (w_nash * s_nash[key] + w_br * s_br[key]) / (w_nash + w_br)
    return res


def run(game, fixed) -> dict:
    """Sweep of p for RNR, then naive mix between the p=0 and p=1 strategies."""
    rnr, strategies = [], []
    for p in P_VALUES:
        trainer = RestrictedNashTrainer(game, np.random.default_rng(0), p=p, fixed=fixed)
        trainer.train(N_ITER, N_ITER)
        profile = {k: s for k, s in trainer.average_strategy().items()
                   if game.current_player(k[1:]) == 0}
        strategies.append(profile)
        rnr.append(measure(game, profile, fixed))
    naive = [measure(game, mix(game, strategies[0], strategies[-1], lam), fixed) for lam in P_VALUES]
    return {"rnr": np.array(rnr), "naive": np.array(naive)}


def plot(game, results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res["rnr"][:, 0], res["rnr"][:, 1], "o-", lw=2, label="restricted Nash response")
        ax.plot(res["naive"][:, 0], res["naive"][:, 1], "o--", lw=1.5, label="mix of Nash and best response")
        ax.axhline(-1/18, color="gray", ls=":", lw=1, label="game value (-1/18)")
        ax.set_xlabel("exploitability of player 0")
        ax.set_ylabel("gain of player 0 against the opponent")
        ax.set_title(f"Opponent who {name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Exploiting a biased opponent without becoming exploitable")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    game = KuhnGame()
    results = {}
    for name, fixed in opponents().items():
        print(f"opponent who {name} ...")
        results[name] = run(game, fixed)
        for p, (expl, gain) in zip(P_VALUES, results[name]["rnr"]):
            print(f"  p = {p:.1f}   exploitability = {expl:.4f}   gain = {gain:.4f}")
    plot(game, results, FIGURES / "rnr.png")
    print("saved figures/rnr.png")


if __name__ == "__main__":
    main()
