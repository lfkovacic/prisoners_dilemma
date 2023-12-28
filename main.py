import random
import numpy as np
from game import PrisonersDilemma
from agents import *

game = PrisonersDilemma()
player1 = MLAgent(input_size=16)
player2 = TitForTat()

#player1.train(game, './ml_agent_model.h5', opponent=player1, games_per_epoch=1, epochs=10, num_rounds=32)

game.play_game(100, player1, player2)
