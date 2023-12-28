import os
import datetime
import random
import numpy as np
import tensorflow as tf


class RandomAgent:
    def make_move(self, history, opponent_id):
        """
        Returns a random move.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        return random.choice(['C', 'D'])

    def update_opponent_move(self, move):
        return 0


class TitForTat:
    def __init__(self):
        self.last_opponent_move = 'C'

    def make_move(self, history, opponent_id):
        """
        Returns the next move based on the Tit for Tat strategy.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        if (history != []):
            self.last_opponent_move = history[-1][opponent_id]
        return self.last_opponent_move

class TitForTwoTats:
    def __init__(self):
        self.defection_count = 0

    def make_move(self, history, opponent_id):
        if history != [] and history[-1][opponent_id] == 'D':
            self.defection_count += 1
        else:
            self.defection_count = 0

        return 'D' if self.defection_count >= 2 else 'C'

class TitForTatWithForgiveness:
    def __init__(self, forgiveness_threshold=1):
        self.forgiveness_threshold = forgiveness_threshold
        self.defection_count = 0

    def make_move(self, history, opponent_id):
        if history != [] and history[-1][opponent_id] == 'D':
            self.defection_count += 1
        else:
            self.defection_count = 0

        return 'D' if self.defection_count <= self.forgiveness_threshold else 'C'

class GradualTitForTat:
    def __init__(self, retaliation_threshold=2):
        self.retaliatory_phase = False
        self.retaliatory_count = 0
        self.retaliatory_threshold = retaliation_threshold

    def make_move(self, history, opponent_id):
        if history != []:
            return 'C'

        if self.retaliatory_phase:
            self.retaliatory_count += 1

        if history[-1][opponent_id] == 'D':
            self.retaliatory_phase = True

        return 'D' if self.retaliatory_phase and self.retaliatory_count <= self.retaliatory_threshold else 'C'



class TatForTit:
    def __init__(self):
        self.last_opponent_move = 'D'

    def make_move(self, history, opponent_id):
        """
        Returns the next move based on the Tit for Tat strategy.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        if (history != []):
            self.last_opponent_move = history[-1][opponent_id]
        if (self.last_opponent_move == 'C'):
            return 'D'
        else:
            return 'C'

    def update_opponent_move(self, move):
        """
        Updates the last opponent move.

        Parameters:
            move (str): Opponent's move ('C' for cooperate, 'D' for defect).
        """
        if (move == 'C'):
            self.last_opponent_move = 'D'
        else:
            self.last_opponent_move = 'C'


class MLAgent:
    def __init__(self, path='./ml_agent_model.h5', input_size=16, nastiness_constant=2, vengefullness_constant=0.8):
        self.input_size = input_size
        self.model_path = path
        self.model = self.build_model()
        self.nastiness_constant = nastiness_constant #lower = more nasty
        self.vengefullness_constant = vengefullness_constant #lower = less vengeful

    def build_model(self):
        if os.path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        else:
            # Define the model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(self.input_size,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Compile the model
            model.compile(optimizer='adam',
                          loss='binary_crossentropy', metrics=['accuracy'])
            return model

    def train(self, game, path, opponent, epochs=10, games_per_epoch=10, num_rounds=10):
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        # Define instances of players 1 and 2
        player1 = self
        player2 = opponent

        player2.__init__()

        for epoch in range(epochs):
            total_X_train = []
            total_y_train = []

            for _ in range(games_per_epoch):
                # Reset the game state for a new self-play game
                game.__init__()

                for round_num in range(num_rounds):
                    # Collect data for training from self-play games
                    history = game.get_history()
                    # print(f"History: {history}")

                    # Get the last 8 rounds' moves
                    last_moves = history[-self.input_size:] if len(history) >= self.input_size else [(
                        'C', 'C', (3, 3))] * (self.input_size - len(history)) + history
                    # print(f"last_moves: {last_moves}")

                    # Extract moves and payoffs
                    moves = [move for move, _, _ in last_moves]
                    payoffs = [payoff for _, _, payoff in last_moves]
                    # Encode opponent's moves
                    encoded_moves = player1.encode_moves(moves)
                    # print(f"payoffs: {payoffs}, epoch: {epoch}")
                    # Calculate mean expected payoff
                    mean_expected_payoffs = np.mean([t[0] for t in payoffs])

                    # Try and predict opponent's next move
                    predicted_opponent_move = player2.make_move(history, 0)
                    print(f"Predicted opponent move: {predicted_opponent_move}")

                    # See what happens if you make a move
                    predicted_agent_move = player1.make_move(history, 1)

                    # Append the predicted moves to the payoffs
                    appended_payoffs = payoffs
                    appended_payoffs.append(game.train_round(
                        predicted_agent_move, predicted_opponent_move))

                    # See what the new mean is
                    predicted_mean_expected_payoff = np.mean(
                        [t[0] for t in payoffs])
                    print(
                        f"Predicted mean payoff: {predicted_mean_expected_payoff}, round: {round_num}, epoch: {epoch}")

                    # Choose the maximally beneficial action based on the agent's expected payoff relative to opponent's
                    if predicted_mean_expected_payoff > self.nastiness_constant:
                        max_beneficial_action = 'D'  # Agent expects positive payoff, defect
                    elif predicted_mean_expected_payoff < self.vengefullness_constant:
                        max_beneficial_action = 'D'  # Opponent consistently defects, defect
                    else:
                        max_beneficial_action = 'C'  # Otherwise, cooperate
                    print(f"Maximally beneficial action: {max_beneficial_action}, round: {round_num}, epoch: {epoch}")

                    # Play the round using the game's play_round method
                    move1 = max_beneficial_action
                    move2 = player2.make_move(history, 0)
                    payoff1, payoff2 = game.play_round(move1, move2)

                    # Update the game history
                    history = game.get_history()

                    # Create training data
                    # Flatten the array
                    total_X_train.append(encoded_moves.flatten())
                    # Use 1 for 'D', 0 for 'C'
                    total_y_train.append(
                        1 if max_beneficial_action == 'D' else 0)

                # Convert to numpy arrays for training
                X_train = np.array(total_X_train)
                y_train = np.array(total_y_train)
                print(f"X_train: {X_train}, epoch: {epoch}")
                print(f"y_train: {y_train}, epoch: {epoch}")

                # Train the model
                player1.model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    callbacks=[tensorboard_callback],
                    initial_epoch=epoch,
                )

        # Save the model after training
        player1.model.save(path)

    def encode_moves(self, moves):
        """
        Encodes the opponent's moves into a binary array.

        Parameters:
            moves (list): List of opponent's moves ('C' for cooperate, 'D' for defect).

        Returns:
            numpy array: Binary array representing the opponent's moves.
        """
        encoded_moves = [1 if move == 'D' else 0 for move in moves]
        return np.array(encoded_moves)

    def make_move(self, history, opponent_id):
        """
        Returns the next move based on the MLAgent's prediction.

        Parameters:
            history (list): List of tuples representing (move1, move2, payoff) for each round.
            opponent_id (int): The index of the opponent in the history tuple.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        opponent_moves = ([move[opponent_id] for move in history[-self.input_size:]])

        if len(opponent_moves) < self.input_size:
            opponent_moves = [
                'C'] * (self.input_size - len(opponent_moves)) + opponent_moves

        encoded_moves = self.encode_moves(opponent_moves[-self.input_size:])
        prediction = self.model.predict(np.array([encoded_moves]))[0][0]

        return 'D' if prediction > 0.5 else 'C'


class JerkFace:
    def make_move(self, history, opponent_id):
        """
        Returns D because he's a jerk
        """
        return 'D'


class MotherTheresa:
    def make_move(self, history, opponent_id):
        """
        Returns C because she's holy
        """
        return 'C'

class Tester:
    def __init__(self, cooperation_rounds=5):
        self.cooperation_rounds = cooperation_rounds
        self.round_count = 0

    def make_move(self, history, opponent_id):
        if self.round_count < self.cooperation_rounds:
            self.round_count += 1
            return 'C'
        else:
            return 'D'

class SmarterTester:
    def __init__(self, cooperation_threshold=0.8, cooperation_rounds=5):
        self.cooperation_threshold = cooperation_threshold
        self.cooperation_rounds = cooperation_rounds
        self.round_count = 0

    def make_move(self, history, opponent_id):
        if not history:
            return 'C'

        opponent_moves = [move[opponent_id] for move in history]

        cooperation_percentage = sum(1 for move in opponent_moves if move == 'C') / len(opponent_moves)

        if cooperation_percentage >= self.cooperation_threshold:
            self.round_count += 1
            return 'C' if self.round_count <= self.cooperation_rounds else 'D'
        else:
            return 'D'
