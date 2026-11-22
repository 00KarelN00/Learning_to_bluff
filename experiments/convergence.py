"""Convergence of regret matching on RPS and chance sampling CFR on Kuhn poker.

    python experiments/convergence.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # packages live at the repo root

from games.rps import RPSGame
from games.kuhn import KuhnGame
from agents.regret_matching import RegretMatcher
from training.rps_training import train
from training.cfr_training import ChanceSamplingCFRTrainer

OUT = ROOT / "figures" / "convergence.png"


def main():
    rng = np.random.default_rng(0)

    # RPS self-play, player 0 starts biased towards rock
    log_every = 10
    agents = [RegretMatcher(3, rng, np.array([1., 0., 0.])), RegretMatcher(3, rng)]
    log = train(RPSGame(), agents, 20_000, log_every=log_every)
    avg = np.array([s[0] for s in log["average_strategy"]])
    x_rps = np.arange(len(avg)) * log_every + 1

    # Kuhn, chance sampling CFR
    eval_every = 200
    trainer = ChanceSamplingCFRTrainer(KuhnGame(), rng)
    values = trainer.train(200_000, eval_every=eval_every)
    x_kuhn = np.arange(1, len(values) + 1) * eval_every

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for i, name in enumerate(RPSGame.ACTIONS):
        ax1.plot(x_rps, avg[:, i], label=name)
    ax1.axhline(1/3, color="gray", ls="--", lw=1, label="Nash (1/3)")
    ax1.set_title("Rock-paper-scissors, regret matching self-play")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("average strategy of player 0")
    ax1.set_ylim(0, 1)
    ax1.set_xscale("log")
    ax1.legend()

    ax2.plot(x_kuhn, values, label="value of average strategies")
    ax2.axhline(-1/18, color="gray", ls="--", lw=1, label="theory (-1/18)")
    ax2.set_title("Kuhn poker, chance sampling CFR")
    ax2.set_xlabel("iterations")
    ax2.set_ylabel("game value for player 0")
    ax2.set_xscale("log")
    ax2.legend()

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
