
import pandas as pd
import datetime


class PrisonersDilemma2:
    def __init__(self, players, output_file=f"game_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", summary_file=f"tournament_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"):
        self.payoff_matrix = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        self.players = players
        self.history = []
        self.total_scores = {player.get_name(): 0 for player in players}
        self.output_file = output_file
        self.summary_file = summary_file
        self.tournament_scores = {
            p1.get_name(): {p2.get_name(): None for p2 in players} for p1 in players}

    def play_game(self, num_rounds):
        for player1 in self.players:
            for player2 in self.players:
                name1 = player1.get_name()
                name2 = player2.get_name()

                scores = self._play_rounds(
                    num_rounds, player1, player2, name1, name2)
                if name1 !=name2:
                    self.tournament_scores[name1][name2] = scores[name1]
                else:
                    self.tournament_scores[name1][name2] = scores[name1]/2

        self._output_to_file()
        self._output_summary_table()

    def _play_rounds(self, num_rounds, player1, player2, name1, name2):
        local_history = []
        local_scores = {name1: 0, name2: 0}
        for _ in range(num_rounds):
            history = self.get_history()

            move1 = player1.make_move(local_history, 1)
            move2 = player2.make_move(local_history, 0)

            # Verify moves and log them for debugging
            if move1 not in ("C", "D") or move2 not in ("C", "D"):
                print(
                    f"Unexpected move detected - Move1: {move1}, Move2: {move2}")
                raise ValueError(
                    f"Invalid move detected from {name1 if move1 not in ('C', 'D') else name2}")

            payoff = self.play_round(move1, move2, name1, name2)
            local_history.append((move1, move2, payoff))
            local_scores[name1] += payoff[0]
            local_scores[name2] += payoff[1]
        self.history.append((name1, name2, local_history, local_scores))
        return local_scores

    def play_round(self, move1, move2, name1, name2):
        payoff = self.payoff_matrix[(move1, move2)]
        self.total_scores[name1] += payoff[0]
        self.total_scores[name2] += payoff[1]
        return payoff

    def get_history(self):
        return self.history

    def get_total_scores(self):
        return self.total_scores

    def _output_to_file(self):
        output_data = []
        for matchup in self.history:
            name1, name2, rounds, scores = matchup
            for round_num, (move1, move2, payoff) in enumerate(rounds, start=1):
                output_data.append({
                    "Player 1": name1,
                    "Player 2": name2,
                    "Round": round_num,
                    "Player 1 Move": move1,
                    "Player 2 Move": move2,
                    "Player 1 Reward": payoff[0],
                    "Player 2 Reward": payoff[1],
                })
            output_data.append({"Total Score Player 1": scores[name1],
                                "Total Score Player 2": scores[name2]})

        df = pd.DataFrame(output_data)
        df.to_excel(self.output_file, index=False)

    def _output_summary_table(self):
        df_summary = pd.DataFrame(self.tournament_scores).fillna(0)
        df_summary.to_excel(self.summary_file, index=True)
