import numpy as np
from games.kuhn import KuhnGame
from agents.regret_matching import InfoSetNode, FixedAgent


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
    

class CFRTrainer:
    def __init__(self, game: KuhnGame, rng):
        self.game = game
        self.nodes: dict[str, InfoSetNode] = {}
        self.rng = rng

    def get_node(self, key: str) -> InfoSetNode:
        """Renvoie le nœud, en le créant au premier passage."""
        if not key in self.nodes.keys():
            self.nodes[key] = InfoSetNode(key, len(self.game.ACTIONS), self.rng)
        return self.nodes[key]

     

    def cfr(self, cards, history: str, reach0: float, reach1: float) -> float:
        """Utilité espérée pour le joueur 0 depuis ce nœud.
        Met à jour regrets et stratégies du nœud au passage."""
        if self.game.is_terminal(history):
            return self.game.payoff(cards, history)
        else: 
            player = self.game.current_player(history)
            key = self.game.infoset_key(cards, history)
            agent = self.get_node(key)
            strategy = agent.get_strategy()
            if player ==0:
                sons_values = np.array([self.cfr(cards, (history + action), reach0*strategy[i], reach1) for i, action in enumerate(self.game.ACTIONS)])
                current_value = sons_values @ strategy
                regret = sons_values - current_value
                agent.update(regret, reach0, reach1)
                
            else:
                sons_values = np.array([self.cfr(cards, (history + action), reach0, reach1*strategy[i]) for i, action in enumerate(self.game.ACTIONS)])
                current_value = sons_values @ strategy
                regret = -(sons_values - current_value)
                agent.update(regret, reach1, reach0)
            return current_value

        
    def train(self, n_iter: int, n_freq) -> list[float]:
        """À chaque itération : cfr sur les 6 donnes, chacune pondérée par 1/6.
        Renvoie la valeur du jeu pour le joueur 0 à chaque itération."""
        values_in_time = []
        for i in range(n_iter):
            current_game_value = 0
            for cards in self.game.deals():
                current_game_value += self.cfr(cards, "", 1,1)
            if (i+1)%n_freq ==0:
                values_in_time.append(current_game_value/6)
        return values_in_time
    
        

    def average_strategy(self) -> dict[str, np.ndarray]: 
        res = {}
        for key, agent in self.nodes.items():
            res[key]=agent.average_strategy()
        return res 


class RestrictedNashTrainer(CFRTrainer):
    def __init__(self, game: KuhnGame, rng, p: float, fixed: dict[str, np.ndarray], fixed_player: int = 1):
        super().__init__(game, rng)
        self.p = p
        self.fixed = fixed
        self.fixed_player = fixed_player
        self.in_fixed_pass = False

    def get_node(self, key: str):
        """During the fixed pass, the opponent's nodes play `fixed` and never learn."""
        player = self.game.current_player(key[1:])
        if self.in_fixed_pass and player == self.fixed_player:
            return FixedAgent(self.fixed[key], self.rng)
        else:
            return super().get_node(key)

    def train(self, n_iter: int, n_freq) -> list[float]:
        """À chaque itération : pour chaque donne, un cfr où l'adversaire est fixe (reach p)
        et un cfr où il s'adapte (reach 1-p)."""
        values_in_time = []
        for i in range(n_iter):
            current_game_value = 0
            for cards in self.game.deals():
                for fixed_pass, weight in [(True, self.p), (False, 1 - self.p)]:
                    self.in_fixed_pass = fixed_pass
                    if self.fixed_player == 1:
                        value = self.cfr(cards, "", 1, weight)
                    else:
                        value = self.cfr(cards, "", weight, 1)
                    current_game_value += weight*value
            if (i+1)%n_freq ==0:
                values_in_time.append(current_game_value/6)
        self.in_fixed_pass = False
        return values_in_time 


