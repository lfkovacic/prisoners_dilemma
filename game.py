
class PrisonersDilemma:
    def __init__(self):
        # Payoff matrix for the Prisoner's Dilemma
        self.payoff_matrix = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        self.history = []
        self.total_scores = {'Player 1': 0, 'Player 2': 0}

    def play_game(self, num_rounds, player1, player2):
        self.history = []
        self.total_scores = {'Player 1': 0, 'Player 2': 0}
        for _ in range(num_rounds):
            history = self.get_history()
            if (history!=[]):                
                print(history[-1])
            move1 = player1.make_move(history, 1)
            move2 = player2.make_move(history, 0)

            payoff1, payoff2 = self.play_round(move1, move2)

        history = self.get_history()

        print("\nGame History:")
        for round_num, (move1, move2, payoff) in enumerate(history, start=1):
            print(f"Round {round_num}: Player 1 move: {move1}, Player 2 move: {move2}, Payoff: {payoff}")

        total_scores = self.get_total_scores()
        print("\nTotal Scores:")
        for player, score in total_scores.items():
            print(f"{player}: {score}")


    def train_round(self, move1, move2):
        payoff = self.payoff_matrix[(move1, move2)]
        return payoff


    def play_round(self, move1, move2):
        """
        Simulates one round of the Prisoner's Dilemma game.

        Parameters:
            move1 (str): Move of player 1 ('C' for cooperate, 'D' for defect).
            move2 (str): Move of player 2 ('C' for cooperate, 'D' for defect).

        Returns:
            tuple: Payoffs for player 1 and player 2.
        """
        payoff = self.payoff_matrix[(move1, move2)]
        self.history.append((move1, move2, payoff))
        self.total_scores['Player 1'] += payoff[0]
        self.total_scores['Player 2'] += payoff[1]
        return payoff

    def get_history(self):
        """
        Returns the history of moves and payoffs.

        Returns:
            list: List of tuples representing (move1, move2, payoff) for each round.
        """
        return self.history

    def get_total_scores(self):
        """
        Returns the total scores for each player.

        Returns:
            dict: Dictionary containing total scores for each player.
        """
        return self.total_scores