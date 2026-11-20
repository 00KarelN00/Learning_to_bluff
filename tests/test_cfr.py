import numpy as np
from games.kuhn import KuhnGame
from training.cfr_training import CFRTrainer


def test_nodes_after_one_iteration():
    """Check if the right nodes have been created after the first iteration"""
    t = CFRTrainer(KuhnGame(), np.random.default_rng(0))
    t.train(1, 1)
    assert len(t.nodes) == 12
    assert set(t.nodes) == {"0", "1", "2", "0p", "1p", "2p",
                            "0b", "1b", "2b", "0pb", "1pb", "2pb"}


def test_cfr_kuhn():
    t = CFRTrainer(KuhnGame(), np.random.default_rng(0))
    values = t.train(5_000, 1)   

    # average value approaches -1/18
    assert abs(np.mean(values) - (-1 / 18)) < 5e-3

    # 2. average strategy looks like the equilibrium
    BET = KuhnGame.ACTIONS.index("b")
    s = {k: v[BET] for k, v in t.average_strategy().items()}
    alpha = s["0"]  
    assert 0 <= alpha <= 1 / 3 + 0.02
    assert s["1"] < 0.03                                  # Q : always p
    assert abs(s["2"] - 3 * alpha) < 0.05                 # K : bets 3*alpha
    assert abs(s["1pb"] - (alpha + 1 / 3)) < 0.05         # Q against a bet: bets with alpha + 1/3
    assert s["0pb"] < 0.03 and s["2pb"] > 0.97
    assert abs(s["0p"] - 1 / 3) < 0.03                    # player 1 : bluffs with J one time out of three
    assert s["1p"] < 0.03 and s["2p"] > 0.97
    assert s["0b"] < 0.03                                 # J against a bet: always folds
    assert abs(s["1b"] - 1 / 3) < 0.03                    # Q against a bet : bets one time out of three
    assert s["2b"] > 0.97                                 # K : alwatys bets