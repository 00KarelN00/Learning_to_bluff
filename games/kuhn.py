import numpy as np


class KuhnGame:
    CARDS = ("queen","king","ace")
    ACTIONS = ("b", "p")
    HISTORIES = ["","p","b","pb"]
    TERMINALS = ["bb","bp","pp","pbp","pbb"]
    SONS = [[2,3],[4,5],[],[],[6,7],[],[],[]]
    N_PLAYERS = 2 

    def deals(self):
        return [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]

    def is_terminal(self, history: str):
        return history in self.TERMINALS

    def current_player(self, history: str):
        return len(history)%2

    def payoff(self, cards: tuple[int, int], history):
        if not self.is_terminal(history): raise ValueError("not terminal")

        bet_count = history.count("b")
        if bet_count ==2:
            return 2 if cards[0]>cards[1] else -2
        if bet_count ==1:
            return 1 if len(history)==2 else -1
        else:
            return 1 if cards[0]>cards[1] else -1

    def infoset_key(self,cards,history):
        return f"{cards[self.current_player(history)]}"+history