"""Trainer-independent evaluation of a Kuhn strategy profile.

A profile is a dict {infoset key: probability vector over game.ACTIONS}.
Missing infosets are played uniformly.
"""
import itertools
import numpy as np


def average_profile(trainer) -> dict[str, np.ndarray]:
    """Average strategy of every infoset a trainer has visited."""
    return {key: node.average_strategy() for key, node in trainer.nodes.items()}


def _value(game, profile, cards, history) -> float:
    if game.is_terminal(history):
        return game.payoff(cards, history)
    n = len(game.ACTIONS)
    strat = profile.get(game.infoset_key(cards, history), np.ones(n) / n)
    return sum(p * _value(game, profile, cards, history + a)
               for p, a in zip(strat, game.ACTIONS) if p > 0)


def expected_value(game, profile) -> float:
    """Exact expected payoff for player 0, averaged over the deals."""
    deals = game.deals()
    return sum(_value(game, profile, cards, "") for cards in deals) / len(deals)


def best_response_value(game, profile, player: int) -> float:
    """Best value `player` can get against the other player's strategy in `profile`.
    Brute force over all pure strategies (2^6 = 64 per player in Kuhn)."""
    n = len(game.ACTIONS)
    histories = [h for h in game.HISTORIES if game.current_player(h) == player]
    keys = [f"{c}{h}" for c in range(len(game.CARDS)) for h in histories]
    sign = 1 if player == 0 else -1
    best = -np.inf
    for choice in itertools.product(range(n), repeat=len(keys)):
        pure = {k: np.eye(n)[a] for k, a in zip(keys, choice)}
        best = max(best, sign * expected_value(game, {**profile, **pure}))
    return best


def exploitability(game, profile) -> float:
    """Average gain of a best response over the profile, 0 at a Nash equilibrium."""
    return (best_response_value(game, profile, 0) + best_response_value(game, profile, 1)) / 2
