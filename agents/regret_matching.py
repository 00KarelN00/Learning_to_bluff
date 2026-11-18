import numpy as np

class FixedAgent:
    """Baseline opponent: always plays the same mixed strategy."""

    def __init__(self, strategy: np.ndarray, rng: np.random.Generator):
        self.strategy = strategy
        self.rng = rng
        self.n_actions = len(strategy)


    def get_strategy(self) -> np.ndarray: return self.strategy
    def sample_action(self) -> int: return self.rng.choice(self.n_actions, p = self.strategy)
    def update(self, regrets: np.ndarray, strategy_weight: float = 1.0, regrets_weight: float = 1.0) -> None:
        pass
    def average_strategy(self): return self.strategy


class RegretMatcher:
    """Learns with regret matching. Only stores two vectors."""

    def __init__(self, n_actions: int, rng: np.random.Generator,
                 initial_regrets: np.ndarray | None = None):
        self.n_actions = n_actions
        self.regret_sum = np.zeros(n_actions) if initial_regrets is None else initial_regrets
        self.strategy_sum = np.zeros(n_actions)
        self.rng = rng

    def get_strategy(self) -> np.ndarray:
        """Current strategy: normalized positive regrets, uniform if they are all <= 0."""   
        pos = np.maximum(self.regret_sum, 0)
        return pos / pos.sum() if pos.sum()>0 else np.ones(self.n_actions)/self.n_actions

    def sample_action(self) -> int:
        """Samples an action from the current strategy.""" 
        return self.rng.choice(self.n_actions, p = self.get_strategy())

    def update(self, regrets: np.ndarray, strategy_weight: float = 1.0, regrets_weight: float = 1.0) -> None:
        """1) adds strategy_weight * current strategy to strategy_sum,
           2) THEN adds regrets to regret_sum.""" 
        self.strategy_sum += self.get_strategy()*strategy_weight
        self.regret_sum += regrets*regrets_weight

    def average_strategy(self) -> np.ndarray:
        """Normalized strategy_sum (uniform if it's zero). This is the final result.""" 
        return np.ones(self.n_actions)/self.n_actions if self.strategy_sum.sum()  == 0 else self.strategy_sum/self.strategy_sum.sum()

class InfoSetNode(RegretMatcher):
    """A RegretMatcher that knows its infoset key."""
    def __init__(self, key: str, n_actions: int, rng): 
        super().__init__(n_actions, rng)
        self.key = key


        