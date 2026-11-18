import numpy as np
from games.kuhn import KuhnGame
from agents.regret_matching import InfoSetNode


class ChanceSamplingCFRTrainer:
    def __init__(self, game: KuhnGame, rng):
        self.game = game
        self.nodes: dict[str, InfoSetNode] = {}
        self.rng = rng

    def get_node(self, key: str) -> InfoSetNode:
        """Returns the node, creates it the first time."""
        if not key in self.nodes.keys():
            self.nodes[key] = InfoSetNode(key, len(self.game.ACTIONS), self.rng) 
        return self.nodes[key]

    def regret(self,cards):
        d = {}
        for h in self.game.TERMINALS:
            d[h] = self.game.payoff(cards,h)
        for h in self.game.HISTORIES[::-1]:
            key = self.game.infoset_key(cards, h)
            agent = self.get_node(key)
            sons_values = np.array([d[(h+action)] for action in self.game.ACTIONS])
            value = agent.get_strategy() @ sons_values
            d[h] = value
        def loop(h,a0,a1):
            if not self.game.is_terminal(h):
                key = self.game.infoset_key(cards,h)
                agent = self.get_node(key)
                strat = agent.get_strategy()
                sons_values = np.array([d[(h+action)] for action in self.game.ACTIONS])
                regret = sons_values - d[h] 
                if self.game.current_player(h)==0:
                    agent.update(regret, a0, a1)
                    for index,action in enumerate(self.game.ACTIONS):
                        loop((h+action), a0*strat[index],a1)
                else:
                    regret = -regret
                    agent.update(regret, a1, a0)
                    for index, action in enumerate(self.game.ACTIONS):
                        loop((h+action), a0, a1*strat[index])
        loop("",1,1)



    def play_round(self):
        cards = self.rng.choice(self.game.deals())
        self.regret(cards= cards)
        
    def game_value(self) -> float:
        """Exact value of the average strategies, averaged over the 6 deals."""
        d = {}
        for cards in self.game.deals():
            d[cards] = {}
            for h in self.game.TERMINALS:
                d[cards][h] = self.game.payoff(cards, h)
            for h in self.game.HISTORIES[::-1]:
                key = self.game.infoset_key(cards, h)
                agent = self.get_node(key)
                sons_values = np.array([d[cards][h + action] for action in self.game.ACTIONS])
                value = agent.average_strategy() @ sons_values
                d[cards][h] = value
        return sum(d[cards][""] for cards in self.game.deals()) / 6



    def train(self, n_iter: int, eval_every) -> list[float]:
        values = []
        for i in range(n_iter):
            self.play_round()
            if (i+1) % eval_every ==0:
                values.append(self.game_value())
        return values
    

        


