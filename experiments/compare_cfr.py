"""Vanilla CFR vs chance sampling CFR on Kuhn poker.

To keep the comparison fair:
- same metric for both: exploitability (and value) of the AVERAGE strategy,
  computed by training/evaluation.py, not by the trainers themselves;
- same budget unit: deals traversed. One vanilla iteration walks the 6 deals,
  one chance sampling iteration walks 1, so both are checkpointed at the same
  numbers of deals. Wall-clock training time (evaluation excluded) is also logged.
Chance sampling is random, so it runs on several seeds; vanilla is deterministic.
Also writes the learned vanilla strategy next to the theoretical equilibrium.

    python experiments/compare_cfr.py [--deals 120000] [--seeds 3]

Outputs figures/cfr_comparison.png and figures/kuhn_strategy.md.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # packages live at the repo root

from games.kuhn import KuhnGame
from training.cfr_training import CFRTrainer, ChanceSamplingCFRTrainer
from training.evaluation import average_profile, expected_value, exploitability

FIGURES = ROOT / "figures"
N_DEALS = len(KuhnGame().deals())
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8984"
KEYS = ["0", "1", "2", "0p", "1p", "2p", "0b", "1b", "2b", "0pb", "1pb", "2pb"]
SITUATIONS = {"": "player 0, first move", "p": "player 1, after a check",
              "b": "player 1, facing a bet", "pb": "player 0, facing a bet"}


def theory(alpha: float) -> dict[str, float]:
    """Equilibrium P(bet) per infoset, for a given alpha in [0, 1/3]."""
    return {"0": alpha, "1": 0, "2": 3 * alpha, "0pb": 0, "1pb": alpha + 1 / 3, "2pb": 1,
            "0p": 1 / 3, "1p": 0, "2p": 1, "0b": 0, "1b": 1 / 3, "2b": 1}


def strategy_table(game, profile: dict[str, np.ndarray]) -> str:
    """Markdown table: learned vs theoretical P(bet) for the 12 infosets,
    alpha taken as the learned P(bet) at "0"."""
    bet = game.ACTIONS.index("b")
    learned = {k: profile[k][bet] for k in KEYS}
    alpha = learned["0"]
    th = theory(alpha)
    lines = [f"alpha (learned P(bet) at \"0\") = {alpha:.4f}", "",
             "| situation | card | learned P(bet) | theoretical P(bet) |",
             "|---|---|---|---|"]
    for k in KEYS:
        lines.append(f"| {SITUATIONS[k[1:]]} | {game.CARDS[int(k[0])]} "
                     f"| {learned[k]:.4f} | {th[k]:.4f} |")
    return "\n".join(lines)


def checkpoints(budget: int, n_points: int = 30) -> np.ndarray:
    """Log-spaced numbers of deals, multiples of 6 so vanilla iterations line up."""
    pts = np.geomspace(N_DEALS, budget, n_points) // N_DEALS * N_DEALS
    return np.unique(pts.astype(int))


def run(trainer, deals_per_iter: int, points: np.ndarray) -> dict:
    game = trainer.game
    log = {"deals": [], "time": [], "expl": [], "value": []}
    done, elapsed = 0, 0.0
    for target in points:
        n_iter = (target - done) // deals_per_iter
        t0 = time.perf_counter()
        trainer.train(n_iter, n_iter)  # 1 internal eval at the end, negligible
        elapsed += time.perf_counter() - t0
        done += n_iter * deals_per_iter
        profile = average_profile(trainer)
        log["deals"].append(done)
        log["time"].append(elapsed)
        log["expl"].append(exploitability(game, profile))
        log["value"].append(expected_value(game, profile))
    return {k: np.array(v) for k, v in log.items()}


def main(argv=None):
    """argv=None reads the command line; pass [] (or options) when calling from a notebook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=120_000, help="budget in deals traversed")
    parser.add_argument("--seeds", type=int, default=3, help="seeds for chance sampling")
    args = parser.parse_args(argv)

    game = KuhnGame()
    points = checkpoints(args.deals)

    print("vanilla CFR ...")
    vanilla_trainer = CFRTrainer(game, np.random.default_rng(0))
    vanilla = run(vanilla_trainer, N_DEALS, points)
    sampled = []
    for seed in range(args.seeds):
        print(f"chance sampling CFR, seed {seed} ...")
        sampled.append(run(ChanceSamplingCFRTrainer(game, np.random.default_rng(seed)), 1, points))

    cs_expl = np.array([r["expl"] for r in sampled])
    cs_value = np.array([r["value"] for r in sampled])
    cs_time = np.mean([r["time"] for r in sampled], axis=0)

    # ---- table
    print(f"\n{'deals':>8} | {'vanilla expl':>12} {'CS expl (median)':>17} | "
          f"{'vanilla value':>13} {'CS value':>9} | {'vanilla s':>9} {'CS s':>7}")
    for i in np.linspace(0, len(points) - 1, 8).astype(int):
        print(f"{vanilla['deals'][i]:>8} | {vanilla['expl'][i]:>12.5f} {np.median(cs_expl[:, i]):>17.5f} | "
              f"{vanilla['value'][i]:>13.5f} {np.median(cs_value[:, i]):>9.5f} | "
              f"{vanilla['time'][i]:>9.2f} {cs_time[i]:>7.2f}")
    print(f"theory: exploitability 0, value {-1/18:.5f}")

    # ---- plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (axes[0], "deals", "exploitability of average strategy", "Exploitability vs deals traversed"),
        (axes[1], "time", "exploitability of average strategy", "Exploitability vs training time"),
        (axes[2], "deals", "game value for player 0", "Game value vs deals traversed"),
    ]
    for ax, x_key, ylabel, title in panels:
        y_key = "value" if "value" in ylabel else "expl"
        cs_y = cs_value if y_key == "value" else cs_expl
        cs_x = cs_time if x_key == "time" else points
        ax.plot(vanilla[x_key], vanilla[y_key], color=BLUE, lw=2, label="vanilla CFR")
        ax.plot(cs_x, np.median(cs_y, axis=0), color=ORANGE, lw=2,
                label=f"chance sampling CFR (median of {args.seeds} seeds)")
        if args.seeds > 1:
            ax.fill_between(cs_x, cs_y.min(axis=0), cs_y.max(axis=0),
                            color=ORANGE, alpha=0.18, lw=0, label="chance sampling min-max")
        if y_key == "value":
            ax.axhline(-1 / 18, color=MUTED, ls="--", lw=1, label="theory (-1/18)")
        else:
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("deals traversed" if x_key == "deals" else "training time (s)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, color=INK)
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Kuhn poker: vanilla vs chance sampling CFR (same metric, same budget)", color=INK)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "cfr_comparison.png", dpi=120)
    plt.close(fig)
    print("saved figures/cfr_comparison.png")

    # ---- learned strategy vs theory
    table = strategy_table(game, vanilla_trainer.average_strategy())
    (FIGURES / "kuhn_strategy.md").write_text(table + "\n", encoding="utf-8")
    print(f"\n{table}\nsaved figures/kuhn_strategy.md")


if __name__ == "__main__":
    main()
