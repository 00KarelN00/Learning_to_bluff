import numpy as np
from games.rps import RPSGame

def test_payoffs():
    g = RPSGame()
    assert g.payoff(1, 0) == (1, -1)    # paper beats rock
    assert g.payoff(0, 1) == (-1, 1)
    assert g.payoff(2, 2) == (0, 0)

def test_zero_sum():
    g = RPSGame()
    for a0 in range(3):
        for a1 in range(3):
            assert sum(g.payoff(a0, a1)) == 0



from agents.regret_matching import RegretMatcher, FixedAgent
from training.rps_training import regrets, train

def test_regrets():
    g = RPSGame()
    # I played rock (0), opponent played paper (1)
    assert np.allclose(regrets(g, 0, 1, player=0), [0, 1, 2])
    # player 1's side: played paper against rock and won
    assert np.allclose(regrets(g, 0, 1, player=1), [-1, 0, -2])

def test_exploits_fixed_opponent():
    rng = np.random.default_rng(0)
    agents = [RegretMatcher(3, rng), FixedAgent(np.array([.4, .3, .3]), rng)]
    train(RPSGame(), agents, 10_000)
    assert agents[0].average_strategy()[1] > 0.95

def test_self_play_reaches_nash():
    rng = np.random.default_rng(0)
    agents = [RegretMatcher(3, rng, np.array([1., 0., 0.])), RegretMatcher(3, rng)]
    train(RPSGame(), agents, 100_000)
    for ag in agents:
        assert np.allclose(ag.average_strategy(), 1/3, atol=0.02)