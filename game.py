
class PrisonersDilemma:
    def __init__(self, player1, player2):
        # Payoff matrix for the Prisoner's Dilemma
        self.payoff_matrix = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        self.history = []
        self.total_scores = {player1.get_name(): 0, player2.get_name(): 0}

    def play_game(self, num_rounds, player1, player2):
        name1 = player1.get_name()
        name2 = player2.get_name()
        if (name1 == name2):
            name1 = f"{name1}1"
            name2 = f"{name2}2"
        self.history = []
        self.total_scores = {name1: 0, name2: 0}
        for _ in range(num_rounds):
            history = self.get_history()
            move1 = player1.make_move(history, 1)
            move2 = player2.make_move(history, 0)

            payoff1, payoff2 = self.play_round(move1, move2, name1, name2)

        history = self.get_history()

        # print("\nGame History:")
        # for round_num, (move1, move2, payoff) in enumerate(history, start=1):
        #     print(f"Round {round_num}: \n\tPlayer 1 move: {move1}, \n\tPlayer 2 move: {move2}, \nPayoff: {payoff}\n")

        total_scores = self.get_total_scores()
        print("\nTotal Scores:")
        for player, score in total_scores.items():
            print(f"{player}: {score}")


    def train_round(self, move1, move2, name1, name2):
    # Use play_round to ensure consistent history update
        return self.play_round(move1, move2, name1, name2)

    def play_round(self, move1, move2, name1, name2):
        if (name1 == name2):
            name1 = f"{name1}1"
            name2 = f"{name2}2"
        payoff = self.payoff_matrix[(move1, move2)]
        self.history.append((move1, move2, payoff))
        self.total_scores[name1] += payoff[0]
        self.total_scores[name2] += payoff[1]
        return payoff

    def get_history(self):
        return self.history

    def get_total_scores(self):
        """
        Returns the total scores for each player.

        Returns:
            dict: Dictionary containing total scores for each player.
        """
        return self.total_scores