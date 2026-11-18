import numpy as np
from games.kuhn import KuhnGame

def test_kuhn_rules():
    g = KuhnGame()
    assert len(g.deals()) == 6
    assert g.payoff((2, 0), "bb") == 2       # K calls, showdown for 2
    assert g.payoff((0, 2), "pp") == -1      # J vs K, showdown for 1
    assert g.payoff((0, 1), "bp") == 1       # player 1 folds
    assert g.payoff((2, 0), "pbp") == -1     # player 0 folds
    assert g.payoff((0, 1), "pbb") == -2
    assert not g.is_terminal("pb")
    assert g.infoset_key((0, 2), "pb") == "0pb"
    assert g.infoset_key((0, 2), "p") == "2p"