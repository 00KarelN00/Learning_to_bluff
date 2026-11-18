import numpy as np

class RPSGame:
    ACTIONS = ["rock", "paper", "scissors"]
    N_PLAYERS = 2 

    def __init__(self, payoff_matrix: np.ndarray | None = None):
        """payoff_matrix[a, b] = payoff of the player who plays a against b.
        Default: the standard matrix (+1 win, -1 loss, 0 draw)."""
        self.n_actions  = len(self.ACTIONS)  
        self.payoff_matrix = np.array([[0,-1, 1],[1,0,-1],[-1,1,0]]) if payoff_matrix is None else payoff_matrix

    def payoff(self, a0: int, a1: int) -> tuple[float, float]:
        """Payoffs (u0, u1) when player 0 plays a0 and player 1 plays a1."""
        return (self.payoff_matrix[a0, a1], -self.payoff_matrix[a0,a1])


