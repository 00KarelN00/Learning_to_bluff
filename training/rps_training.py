import numpy as np

def regrets(game, a0: int, a1: int, player: int) -> np.ndarray:
    """Regret of `player` for each alternative action, opponent's action fixed.
    Only uses game.payoff (no matrix here).""" 
    potential_gains = np.array([[game.payoff(i,a1)[0] for i in range(game.n_actions)],[game.payoff(a0,i)[1] for i in range(game.n_actions)]])
    return potential_gains[player] - game.payoff(a0,a1)[player]


def play_round(game, agents) -> tuple[int, int]:
    """One round: sample actions, then update each agent. Returns (a0, a1)."""
    a0, a1 = agents[0].sample_action(), agents[1].sample_action()
    regret_0 = regrets(game, a0, a1, 0)
    regret_1 = regrets(game, a0, a1, 1)
    agents[0].update(regret_0), agents[1].update(regret_1)
    return a0, a1


def train(game, agents, n_iter: int, log_every: int = 100) -> dict:
    """Runs n_iter rounds. Returns a log: played moves, plus current
    and average strategy of each agent every log_every iterations."""
    d = {
        "played_moves": [],
        "current_strategy": [],
        "average_strategy": []
    }

    for i in range(n_iter):        
        d["played_moves"].append(play_round(game, agents))
        if i % log_every == 0:
            d["current_strategy"].append((agents[0].get_strategy(), agents[1].get_strategy()))
            d["average_strategy"].append((agents[0].average_strategy(), agents[1].average_strategy()))
    return d 
