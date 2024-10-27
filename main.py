import random
import numpy as np
from game import PrisonersDilemma
from agents import *

# print("\n\nMLAgent vs")
# game = PrisonersDilemma()
# player1 = MLAgent(input_size=16)
# player2 = SmarterTester()

# game.play_game(100, player1, player2)

# player1 = TitForTat()
# player2 = RandomAgent()

# print("\n\nTit for Tat vs")
# game.play_game(100, player1, player2)

player = MLAgent2()
opponents = [
    RandomAgent(),
    TitForTat(),
    TitForTwoTats(),
    TatForTit(),
    player,
    JerkFace(),
    MotherTheresa()
]
for x in range(10):
    opponent = random.choice(opponents)
    player.train(
        game=PrisonersDilemma(player, opponent),
        path="E:\workspace\prisoners_dilemma\ml_agent_model.h5",
        opponent=opponent,
        epochs=10,
        games_per_epoch=1,
        num_rounds=200
    )

# for opponent in opponents:
#     game.play_game(100, player, opponent)

# for opponent1 in opponents:
#     for opponent2 in opponents:
#         game.play_game(100, opponent1, opponent2)