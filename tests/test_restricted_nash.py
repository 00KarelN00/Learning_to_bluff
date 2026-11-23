import numpy as np
from games.kuhn import KuhnGame
from training.cfr_training import RestrictedNashTrainer
from training.evaluation import best_response_value, expected_value


def folds_too_much() -> dict[str, np.ndarray]:
    """Player 1 at equilibrium, except that he folds the king facing a bet (equilibrium: calls 1/3)."""
    bet = {"0p": 1 / 3, "1p": 0, "2p": 1,
           "0b": 0, "1b": 0, "2b": 1}
    return {k: np.array([b, 1 - b]) for k, b in bet.items()}


def test_p0_is_unexploitable():
    """p = 0: the opponent always adapts, so player 0 learns a Nash strategy."""
    game = KuhnGame()
    t = RestrictedNashTrainer(game, np.random.default_rng(0), p=0, fixed=folds_too_much())
    t.train(2_000, 2_000)
    # best value player 1 can get against player 0, 1/18 at equilibrium
    assert best_response_value(game, t.average_strategy(), 1) - 1 / 18 < 1e-2


def test_p1_is_best_response():
    """p = 1: the opponent is always fixed, so player 0 learns a best response to it."""
    game = KuhnGame()
    fixed = folds_too_much()
    t = RestrictedNashTrainer(game, np.random.default_rng(0), p=1, fixed=fixed)
    t.train(2_000, 2_000)
    gain = expected_value(game, {**t.average_strategy(), **fixed})
    best = best_response_value(game, fixed, 0)
    assert best > -1 / 18 + 0.05          # the bias is really exploitable
    assert abs(gain - best) < 1e-2
