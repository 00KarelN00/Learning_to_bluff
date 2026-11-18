# learning-bluff

Learning to bluff with regret minimization.

I'm implementing from scratch the algorithms used to find Nash equilibria in
imperfect information games. Started with rock-paper-scissors, now Kuhn poker.

## What's in there

- **Rock-paper-scissors + regret matching**: the agent exploits a fixed opponent,
  and in self-play the average strategy converges to (1/3, 1/3, 1/3).
- **Kuhn poker + chance sampling CFR**: game value converges to -1/18 and the
  strategies match the known equilibrium (queen bluffs, ace bets 3*alpha, etc.).

```
games/      rps.py, kuhn.py (rules, payoffs, infoset keys)
agents/     regret_matching.py (RegretMatcher, FixedAgent, InfoSetNode)
training/   rps_training.py, cfr_training.py
tests/      pytest tests (game rules + convergence)
figures/    plots
```

## Setup

Python 3.10+

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

The CFR test on Kuhn runs 200k iterations, takes ~30s. To see the learned
strategy for each infoset:

```bash
python -m pytest tests/test_chance_sampling_cfr.py -s
```

## Kuhn poker

3 cards (queen < king < ace), one card each, ante of 1.
Each player can pass/check (`p`) or bet/call (`b`).
An infoset key like `"0pb"` means: I hold the queen, history is check then bet.

At equilibrium player 1 (the second player) wins 1/18 per hand on average, so
going first is a disadvantage even with perfect play.
