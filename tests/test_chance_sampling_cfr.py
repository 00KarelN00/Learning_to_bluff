import numpy as np
from games.kuhn import KuhnGame
from training.cfr_training import ChanceSamplingCFRTrainer


def test_chance_sampling_cfr_converges_to_kuhn_equilibrium():
    rng = np.random.default_rng(0)
    game = KuhnGame()
    trainer = ChanceSamplingCFRTrainer(game, rng)

    values = trainer.train(200_000, eval_every=10_000)
    print("Value during training:", np.round(values, 4))
    print(f"Final value: {values[-1]:.5f}   (theory: {-1/18:.5f})\n")

    # Game value close to -1/18
    assert abs(values[-1] - (-1 / 18)) < 0.01

    # Average strategies, P(bet) = index 0 since ACTIONS = ("b", "p")
    s = {key: node.average_strategy() for key, node in trainer.nodes.items()}
    bet = lambda key: s[key][0]
    names = {0: "queen", 1: "king", 2: "ace"}
    for key in sorted(s):
        print(f"{names[int(key[0])]:>5} | history '{key[1:]}' -> P(b) = {bet(key):.3f}")

    # Checks on the known Kuhn equilibrium
    alpha = bet("0")
    tol = 0.05
    assert -tol <= alpha <= 1 / 3 + tol                  # 0 <= alpha <= 1/3
    assert abs(bet("2") - 3 * alpha) < tol               # ace opens with a bet ~ 3*alpha
    assert bet("1") < tol                                # king never opens with a bet
    assert abs(bet("1b") - 1 / 3) < tol                  # king calls a bet ~ 1/3
    assert abs(bet("1pb") - (alpha + 1 / 3)) < tol       # king calls after p-b ~ alpha+1/3
    assert abs(bet("0p") - 1 / 3) < tol                  # queen bluffs after a check ~ 1/3
