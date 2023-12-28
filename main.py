import random
import numpy as np
from game import PrisonersDilemma
from agents import *

print("\n\nMLAgent vs")
game = PrisonersDilemma()
player1 = MLAgent(input_size=16)
player2 = RandomAgent()

# player1.train(game, './ml_agent_model.h5', opponent=player1, games_per_epoch=1, epochs=3, num_rounds=10)

game.play_game(100, player1, player2)

player1 = TitForTat()
player2 = RandomAgent()

print("\n\nTit for Tat vs")
game.play_game(100, player1, player2)

