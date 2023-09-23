###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import argparse

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def init(n_pop, n_vars):
    return np.random.normal(0, 1, (n_pop, n_vars))

def crossover(pop, fitness, p_mutation):

    n_pop = pop.shape[0]

    for i in range(n_pop):
        p1, p2 = np.random.randint(0, pop.shape[0], size = 2)
        
        alpha = np.random.rand()

        offspring = alpha * pop[p1] + (1 - alpha) * pop[p2] + (np.random.rand(pop[p1].shape[0]) if np.random.rand() < p_mutation else 0)

        pop = np.vstack((pop, offspring))
    
    return pop

def select(n_pop, pop, fitness):

    index = np.argpartition(fitness, n_pop)[-n_pop:]
    
    return pop[index], fitness[index]
    

def train(Continue):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
    experiment_name = 'solution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name = experiment_name,
                    enemies = [2],
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "fastest",
                    visuals = False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    n_pop = 100
    p_mutation = 0.2

    if(os.path.exists(experiment_name + '/pop.bin') and Continue):
        pop = np.fromfile(experiment_name, dtype = np.float64).reshape(n_pop, n_vars)
    else:
        pop = init(n_pop, n_vars)

    fitness = evaluate(env, pop)

    epoch = 30
    best_f = -1
    for i in range(epoch):
        pop = crossover(pop, fitness, p_mutation)
        fitness = evaluate(env, pop)
        pop, fitness = select(n_pop, pop, fitness)

        index_best = np.argmax(fitness)
        pop_best = pop[index_best]
        fitness_best = fitness[index_best]

        # import pdb; pdb.set_trace()

        print('epoch {} best fitness {}'.format(i, fitness_best))

        if(fitness_best > best_f):
            print('best solution saved to {}/best.bin and {}/pop.bin'.format(experiment_name, experiment_name))
            pop_best.tofile(experiment_name + '/best.bin')
            pop.tofile(experiment_name + '/pop.bin')
            best_f = fitness_best


def test(enemy_number):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10

    experiment_name = 'solution'
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name = experiment_name,
                    enemies = [enemy_number],
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "normal",
                    visuals = False)

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    pop = np.fromfile(experiment_name + '/best.bin', dtype = np.float64).reshape(1, n_vars)

    evaluate(env, pop)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'train')
    parser.add_argument('-n', '--enemy_number', type = int, default = 2)
    parser.add_argument('-c', '--Continue', type = bool, default = True)
    
    args = parser.parse_args()

    if(args.mode == 'train'):
        train(args.Continue)
    else:
        test(args.enemy_number)