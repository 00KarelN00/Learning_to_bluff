import numpy as np
from games.kuhn import KuhnGame
from training.evaluation import expected_value, exploitability


def equilibrium(alpha: float) -> dict[str, np.ndarray]:
    """Known Kuhn equilibrium, P(bet) per infoset (ACTIONS = ("b", "p"))."""
    bet = {"0": alpha, "1": 0, "2": 3 * alpha,
           "0pb": 0, "1pb": alpha + 1 / 3, "2pb": 1,
           "0p": 1 / 3, "1p": 0, "2p": 1,
           "0b": 0, "1b": 1 / 3, "2b": 1}
    return {k: np.array([b, 1 - b]) for k, b in bet.items()}


def test_equilibrium_is_unexploitable():
    game = KuhnGame()
    for alpha in (0, 1 / 6, 1 / 3):
        profile = equilibrium(alpha)
        assert abs(expected_value(game, profile) - (-1 / 18)) < 1e-9
        assert abs(exploitability(game, profile)) < 1e-9


def test_uniform_is_exploitable():
    assert exploitability(KuhnGame(), {}) > 0.1
