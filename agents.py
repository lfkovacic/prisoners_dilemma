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
    def __init__(self, input_size=8, model_path='./ml_agent_model.h5'):
        self.input_size = input_size
        self.model_path = model_path
        self.model = self.build_model()

    def build_model(self):
        if os.path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        else:
            # Define the model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # Compile the model
            model.compile(optimizer='adam',
                          loss='binary_crossentropy', metrics=['accuracy'])
            return model

    def train(self, games_per_epoch=10, epochs=10):
        """
        Train the model with given training data.

        Parameters:
            X_train (numpy array): Input training data.
            y_train (numpy array): Target training data.
            epochs (int): Number of epochs for training.
        """
        log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
        for epoch in range(epochs):
            # Collect data for training from self-play games
            inputs = []
            labels = []

            for _ in range(games_per_epoch):
                # Initialize the game state (last 8 decisions of the opponent)
                game_state = np.zeros((1, 8))

                # Play the game for a fixed number of steps
                for _ in range(10):  # You can adjust the number of steps per game
                    print(f"Game state: {game_state}")
                    # Get the model's decision based on the current game state
                    prediction = self.model.predict(game_state)[0][0]
                    print(prediction)
                    binary_decision = np.random.choice(
                        [1, 0], p=[prediction, 1-prediction])

                    # Update the game state with the model's decision
                    game_state = np.concatenate(
                        [game_state[:, 1:], [[binary_decision]]], axis=1)

                reward = 0 if np.sum(game_state) <= 3 else 1

                # Append the game state and the corresponding reward to the training data
                inputs.append(np.squeeze(game_state))
                labels.append(reward)

            inputs = np.array(inputs)
            labels = np.array(labels)

            # Train the model on the collected data
            self.model.fit(inputs, labels, epochs=1, verbose=0)
        self.model.save(self.model_path)

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
        opponent_moves = ([move[opponent_id] for move in history[-8:]])
        print(opponent_moves)
        """
        Returns the next move based on the MLAgent's prediction.

        Parameters:
            opponent_moves (list): List of opponent's moves.

        Returns:
            str: 'C' for cooperate, 'D' for defect.
        """
        if len(opponent_moves) < self.input_size:
            opponent_moves = [
                'C'] * (self.input_size - len(opponent_moves)) + opponent_moves

        encoded_moves = self.encode_moves(opponent_moves[-self.input_size:])
        print(encoded_moves)
        prediction = self.model.predict(np.array([encoded_moves]))[0][0]
        print(prediction)

        return 'D' if np.random.choice([1, 0], p=[prediction, 1-prediction]) == 1 else 'C'


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
