import numpy as np
from agents.regret_matching import RegretMatcher

rng = np.random.default_rng(0)

def test_strategy_from_regrets():
    rm = RegretMatcher(3, rng)
    assert np.allclose(rm.get_strategy(), [1/3, 1/3, 1/3])
    rm.regret_sum = np.array([2., -1., 1.])
    assert np.allclose(rm.get_strategy(), [2/3, 0, 1/3])
    rm.regret_sum = np.array([-5., -1., -2.])
    assert np.allclose(rm.get_strategy(), [1/3, 1/3, 1/3])

def test_update_order():
    rm = RegretMatcher(3, rng)
    rm.update(np.array([0., 1., 2.]))
    assert np.allclose(rm.strategy_sum, [1/3, 1/3, 1/3])   # strategy from BEFORE
    assert np.allclose(rm.get_strategy(), [0, 1/3, 2/3])

def test_average_is_distribution():
    rm = RegretMatcher(3, rng)
    for r in ([1., 0., 0.], [0., 3., 0.], [0., 0., 1.]):
        rm.update(np.array(r))
    assert np.isclose(rm.average_strategy().sum(), 1.0)