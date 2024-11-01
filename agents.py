import tensorflow as tf
import numpy as np
import random
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

physical_devices = tf.config.list_physical_devices('GPU')


class RandomAgent:
    def make_move(self, history, opponent_id):
        """
        Returns a random move.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        return random.choice(['C', 'D'])
    def get_name(self): return "RandomAgent"


class TitForTat:
    def __init__(self):
        self.last_opponent_move = 'C'

    def make_move(self, history, opponent_id):
        """
        Returns the next move based on the Tit for Tat strategy.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        if history == []:
            self.last_opponent_move = 'C'
        else:
            self.last_opponent_move = history[-1][opponent_id]
        return self.last_opponent_move

    def get_name(self):
        return "TitForTat"


class TitForTwoTats:
    def __init__(self):
        self.defection_count = 0

    def make_move(self, history, opponent_id):
        if history != [] and history[-1][opponent_id] == 'D':
            self.defection_count += 1
        else:
            self.defection_count = 0

        return 'D' if self.defection_count >= 2 else 'C'
    def get_name(self): return "TitForTwoTats"


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
    def get_name(self): return "TitForTatWithForgiveness"


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
    def get_name(self): return "GradualTitForTat"


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
    def get_name(self): return "TatForTit"


class MLAgent2:
    def __init__(self, path='./ml_agent_2_model.h5', input_size=16):
        self.last_epoch = 0
        self.model_path = path
        self.input_size = input_size * 2  # Adjusted input size for player and opponent moves
        self.model = self.build_model()

    def build_model(self):
        if os.path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(self.input_size,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def custom_reward(self, move1, move2):
        # Reward logic
        if move1 == 'C' and move2 == 'C':
            return 3
        elif move1 == 'D' and move2 == 'D':
            return 1
        elif move1 == 'C' and move2 == 'D':
            return 0
        else:  # move1 == 'D' and move2 == 'C'
            return 5

    def train(self, game, path, opponent, epochs=10, games_per_epoch=10, num_rounds=10):
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        total_payoff = 0
        player1 = self
        player2 = opponent
        player2.__init__()

        for current_epoch in range(self.last_epoch, self.last_epoch + epochs):
            for _ in range(games_per_epoch):
                game.__init__(player1, player2)
                total_X_train = []
                total_y_train = []
                total_payoff = 0
                for round_num in range(num_rounds):
                    history = game.get_history()
                    last_moves = history[-(self.input_size // 2):] if len(history) >= self.input_size // 2 else [('C', 'C', (3, 3))] * (self.input_size // 2 - len(history)) + history
                    
                    # Extract player and opponent moves from last_moves
                    player_moves = [move[0] for move in last_moves[-(self.input_size // 2):]]
                    opponent_moves = [move[1] for move in last_moves[-(self.input_size // 2):]]

                    # Encode both player and opponent moves
                    encoded_moves = self.encode_moves(player_moves) + self.encode_moves(opponent_moves)

                    predicted_opponent_move = player2.make_move(history, 0)
                    predicted_agent_move = player1.make_move(history, 1)
                    move1 = predicted_agent_move
                    move2 = predicted_opponent_move

                    payoffs = game.train_round(move1, move2, player1.get_name(), player2.get_name())
                    current_payoff = payoffs[0]
                    total_payoff += current_payoff

                    total_X_train.append(encoded_moves)
                    total_y_train.append(total_payoff)

                X_train = np.array(total_X_train)
                y_train = np.array(total_y_train)

                player1.model.fit(
                    X_train,
                    y_train,
                    epochs=current_epoch + epochs,
                    callbacks=[tensorboard_callback],
                    initial_epoch=current_epoch,
                )
        self.last_epoch += epochs
        player1.model.save(path)

    def encode_moves(self, moves):
        return [1 if move == 'D' else 0 for move in moves]

    def make_move(self, history, opponent_id):
        # Extract the last moves for both players
        player_moves = [move[0] for move in history[-(self.input_size // 2):]]
        opponent_moves = [move[1] for move in history[-(self.input_size // 2):]]

        if len(opponent_moves) < self.input_size // 2:
            opponent_moves = ['C'] * (self.input_size // 2 - len(opponent_moves)) + opponent_moves

        if len(player_moves) < self.input_size // 2:
            player_moves = ['C'] * (self.input_size // 2 - len(player_moves)) + player_moves

        # Encode both player and opponent moves
        encoded_player_moves = self.encode_moves(player_moves[-(self.input_size // 2):])
        encoded_opponent_moves = self.encode_moves(opponent_moves[-(self.input_size // 2):])

        # Combine player and opponent moves for prediction
        prediction_input = np.array(encoded_player_moves + encoded_opponent_moves)
        prediction = self.model.predict(np.array([prediction_input]), verbose=0)[0][0]

        return 'D' if prediction > 0.5 else 'C'

    def get_name(self): 
        return "MLAgent"


class MLAgent:
    def __init__(self, path='./ml_agent_model.h5', input_size=16):
        self.current_epoch = 0
        self.input_size = input_size
        self.model_path = path
        self.model = self.build_model()

    def build_model(self):
        if os.path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        else:
            # Define the model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    16+3, activation='relu', input_shape=(self.input_size,)),
                tf.keras.layers.Dense(
                    32, activation='relu', input_shape=(self.input_size, )),
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

        for epoch in range(self.current_epoch, self.current_epoch+epochs):
            total_X_train = []
            total_y_train = []

            for _ in range(games_per_epoch):
                # Reset the game state for a new self-play game
                game.__init__(player1, player2)

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
                    # print(f"Predicted opponent move: {predicted_opponent_move}")

                    # See what happens if you make a move
                    predicted_agent_move = player1.make_move(history, 1)

                    # Append the predicted moves to the payoffs
                    appended_payoffs = payoffs
                    appended_payoffs.append(game.train_round(
                        predicted_agent_move, predicted_opponent_move))

                    # See what the new mean is
                    predicted_mean_expected_payoff = np.mean(
                        [t[0] for t in payoffs])
                    # print(
                    # f"Predicted mean payoff: {predicted_mean_expected_payoff}, round: {round_num}, epoch: {epoch}")

                    # Play the round using the game's play_round method
                    move1 = predicted_agent_move
                    move2 = player2.make_move(history, 0)
                    payoff1, payoff2 = game.play_round(move1, move2)

                    # Update the game history
                    history = game.get_history()

                    # Create training data
                    # Flatten the array
                    total_X_train.append(encoded_moves.flatten())
                    # Use 1 for 'D', 0 for 'C'
                    total_y_train.append(
                        1 if predicted_agent_move == 'D' else 0)

                # Convert to numpy arrays for training
                X_train = np.array(total_X_train)
                y_train = np.array(total_y_train)

                # Train the model
                player1.model.fit(
                    X_train,
                    y_train,
                    epochs=self.current_epoch+epochs,
                    callbacks=[tensorboard_callback],
                    initial_epoch=self.current_epoch,
                )
                self.current_epoch += epochs
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
        opponent_moves = ([move[opponent_id]
                          for move in history[-self.input_size:]])

        if len(opponent_moves) < self.input_size:
            opponent_moves = [
                'C'] * (self.input_size - len(opponent_moves)) + opponent_moves

        encoded_moves = self.encode_moves(opponent_moves[-self.input_size:])
        prediction = self.model.predict(
            np.array([encoded_moves]), verbose=0)[0][0]

        return 'D' if prediction > 0.5 else 'C'


class JerkFace:
    def make_move(self, history, opponent_id):
        """
        Returns D because he's a jerk
        """
        return 'D'
    def get_name(self): return "JerkFace"


class MotherTheresa:
    def make_move(self, history, opponent_id):
        """
        Returns C because she's holy
        """
        return 'C'
    def get_name(self): return "MotherTheresa"


class Tester:
    def __init__(self, history, cooperation_rounds=5):
        self.cooperation_rounds = cooperation_rounds
        self.round_count = 0
        self.opponent_retaliated = False

    def make_move(self, history, opponent_id):
        if self.round_count < self.cooperation_rounds:
            self.round_count += 1
            return 'C'
        else:
            if self.round_count == self.cooperation_rounds:
                return 'D'
            if self.round_count == self.cooperation_rounds+1:
                self.opponent_retaliated = history[-1][opponent_id] == 'D'
            if self.opponent_retaliated:
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

        cooperation_percentage = sum(
            1 for move in opponent_moves if move == 'C') / len(opponent_moves)

        if cooperation_percentage >= self.cooperation_threshold:
            self.round_count += 1
            return 'C' if self.round_count <= self.cooperation_rounds else 'D'
        else:
            return 'D'
