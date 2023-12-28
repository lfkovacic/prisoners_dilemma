import random
import numpy as np
from game import PrisonersDilemma
from agents import JerkFace, MLAgent, MotherTheresa, TatForTit, TitForTat, RandomAgent

game = PrisonersDilemma()
player1 = TatForTit()
player2 = MLAgent()

player2.train(games_per_epoch=10, epochs=10)

game.play_game(100, player1, player2)



