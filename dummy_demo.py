################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name = experiment_name,
                enemies=[2],
                playermode = "ai",
                player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                enemymode = "static",
                level = 2,
                speed = "fastest")

env.play(pcont = np.random.rand(265))

