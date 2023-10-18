###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

import numpy

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import argparse
import pickle
import pandas as pd
from time import sleep
from deap import base
import random
import csv
from random import shuffle
DEBUG_T = 1
RESERVE_Best = 5
# runs simulation
def simulation(env,x):
    f, p, e, t = env.play(pcont = x)
    #return f, p, e, t
    f = 0.8 * (100 - e) + 0.2 * p
    f = f - 0.01 * t  # numpy.log(t)
    return f, p, e, t
def simulation_test(env,x):
    f, p, e, t = env.play(pcont = x)
    return f, p, e, t

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation_test(env,y), x)))
def evaluate_test(env, x):
    return np.array(list(map(lambda y: simulation_test(env,y), x)))

def int2list(enemy_number):
    s = str(enemy_number)
    tmp = []
    for item in s:
        tmp.append(int(item))
    return tmp

def init(n_pop, n_vars):
    return np.random.normal(0, 1, (n_pop, n_vars))

def tournament(pop, fitness):

    p1, p2 = np.random.randint(0, pop.shape[0], size = 2)

    return p1 if fitness[p1] > fitness[p2] else p2

def crossover(env, pop, fitness, p_mutation, selection):

    n_pop = pop.shape[0]

    pop_new = pop

    for i in range(n_pop):
        if(selection == 'random'):
            p1, p2 = np.random.randint(0, pop.shape[0], size = 2)
        elif(selection == 'tournament'):
            p1, p2 = tournament(pop, fitness), tournament(pop, fitness)
        elif(selection == 'DE'):
            while(True):
                p1, p2 = np.random.randint(0, pop.shape[0], size = 2)
                if(p1 != i and p2 != i):
                    break

        alpha = np.random.rand()

        if(selection == 'DE'):
            x = pop[i]
            v = pop[p1] - pop[p2]
            u = x + alpha * v
            l = [x, u, v]
            f = evaluate(env, l)[:, 0]
            offspring = l[np.argmax(f)]
        else: 
            offspring = alpha * pop[p1] + (1 - alpha) * pop[p2] + (np.random.rand(pop[p1].shape[0]) if np.random.rand() < p_mutation else 0)

        pop_new = np.vstack((pop_new, offspring))
    
    return pop_new

def select(n_pop, pop, fitness):

    index = np.argpartition(fitness, n_pop)[-n_pop:]
    
    return pop[index], fitness[index]


def generalist_train(experiment_name, enemies_in_group, selection, p_mutation = 0.2, mode=None, pop1=None, pop2=None):
    if selection is not None:
        experiment_name = experiment_name + "_" + selection
    experiment_name = experiment_name + "_p_" + str(p_mutation)

    n_hidden_neurons = 10
    if pop2 is None:
        pop2 = []
    if pop1 is None:
        pop1 = []
    print("selection is "+str(selection)+"\nmode:"+str(mode))
    selections = [ "DE","random"]
    # initializes simulation in individual evolution mode, for single static enemy.
    if mode == "two":
        if not os.path.exists(experiment_name + "_" + selections[0]):
            os.makedirs(experiment_name + "_" + selections[0])

        if os.path.exists(experiment_name + "_" + selections[0] + '/results.csv'):
            os.remove(experiment_name + "_" + selections[0] + '/results.csv')

        if os.path.exists(experiment_name + "_" + selections[0] + '/best.txt'):
            os.remove(experiment_name + "_" + selections[0] + '/best.txt')
        env = Environment(
            experiment_name=experiment_name + "_" + selections[0],
            enemies=enemies_in_group,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
        )
    else:
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        if os.path.exists(experiment_name + '/results.csv'):
            os.remove(experiment_name + '/results.csv')

        if os.path.exists(experiment_name + '/best.txt'):
            os.remove(experiment_name + '/best.txt')
        env = Environment(
            experiment_name=experiment_name,
            enemies=enemies_in_group,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
        )

    # GLOBAL VARIABLES
    toolbox = base.Toolbox()
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    n_pop = 100
    epoch = 300

    # DATA
    genlist = []
    bestlist = []
    meanlist = []
    stdlist = []
    winner = {"solution": [], "fitness": -200}
    data_fitness = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_player_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_enemy_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_time = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_bestfit = {'fitness': np.array([]), 'player_hp': np.array([]), 'enemy_hp': np.array([]), 'time': np.array([])}

    def init_pop(n_pop, n_vars):
        random.seed(42)
        return np.random.normal(0, 1, (n_pop, n_vars))
    def init_pop_uniform(n_pop, n_vars):
        return np.random.uniform(-1, 1, (n_pop, n_vars))

    def normal_noise(num_vars, p_mutations):
        noise = np.random.normal(0, 1, n_vars)
        for i in range(num_vars):
            if noise[i] > p_mutations:
                noise[i] = p_mutations
            elif noise[i] < - p_mutations:
                noise[i] = - p_mutations
        return noise
    def uniform_noise(num_vars, p_mutations):
        return np.random.uniform(-p_mutations, p_mutations, num_vars)

    def random_noise(num_vars, p_mutations):
        return uniform_noise(num_vars, p_mutations)

    def tournament(pop, fitness, k):
        parents = np.random.randint(0, pop.shape[0], size=k)
        pn = np.argmax(parents)
        return pn

    def tournament_select(pop, k, fitness):
        chosen = []
        for i in range(k):
            chosen = np.append(chosen, tournament(pop, fitness, k))
        # if (DEBUG_T == 1):
        #    print(chosen[0])
        return chosen

    def tournament_322(p1, p2, p3, fitness):
        if fitness[p1] > fitness[p2]:
            return p1, p2 if fitness[p2] > fitness[p3] else p1, p3
        else:
            return p1, p2 if fitness[p1] > fitness[p3] else p2, p3

    def tournament_221(p1, p2, pop, fitness):
        return p1 if fitness[p1] > fitness[p2] else p2

    def cross_two_point(ind1, ind2):
        """Executes a two-point crossover on the input :term:`sequence`
                    individuals. The two individuals are modified in place and both keep
                    their original length.


                    :param ind1: The first individual participating in the crossover.
                    :param ind2: The second individual participating in the crossover.
                    :returns: A tuple of two individuals.

                    This function uses the :func:`~random.randint` function from the Python
                    base :mod:`random` module.
                    """
        size = min(len(ind1), len(ind2))
        # if (DEBUG_T == 1): print("size: ", size)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        return ind1, ind2

    def shuffle_tournament_survivor(n_pop, pop, fitness, player_hp, enemy_hp, time):
        new_pop = []
        new_fitness = []  # = fitness[:]
        new_php = []  # player_hp[:]
        new_ehp = []  # enemy_hp[:]
        new_time = []  # time[:]
        index_pop = random.sample(range(0, np.shape(pop)[0]), np.shape(pop)[0])
        if DEBUG_T == 1:
            pass  # print(index_pop, "\n", np.shape(index_pop), np.shape(pop))
        for i in range(n_pop):
            if i == 0:
                pn = np.argmax(fitness)
                new_pop = [pop[pn]]
            else:
                p1 = index_pop[i]
                p2 = index_pop[i + n_pop]
                pn = tournament_221(p1, p2, pop, fitness)
                if selection == "DE":
                    pn = tournament_221(pn, index_pop[i + 2 * n_pop], pop, fitness)
                l = [pop[pn]]
                new_pop = np.vstack((new_pop, l))
            l2 = [fitness[pn], player_hp[pn], enemy_hp[pn], time[pn]]
            new_fitness.append(l2[0])
            new_php.append(l2[1])
            new_ehp.append(l2[2])
            new_time.append(l2[3])
            # print("shape of result ", np.shape(new_pop))
            # print("shape of fit result ", np.shape(fitness), np.shape(new_fitness))

        return new_pop, new_fitness, new_php, new_ehp, new_time
    def round_robin_tournament(p1, k, pop, fitness):
        contestants = np.random.randint(0, pop.shape[0], size=k+1)
        win = 0
        for contestant in contestants:
            if p1 == contestant:
                continue
            if fitness[p1] >= fitness[contestant]:
                win += 1
        return win
    def round_robin_tournament_survivor(n_pop, pop, fitness, player_hp, enemy_hp, time):
        wins = []
        for i in range(np.shape(pop)[0]):
            wins.append(round_robin_tournament(i, 10, pop, fitness))
        wins_arg = np.argsort(wins)[-n_pop:]
        new_pop = pop[wins_arg] #best n pop winners
        new_fitness = [wins_arg]  # = fitness[:]
        new_php = [wins_arg]  # player_hp[:]
        new_ehp = [wins_arg]  # enemy_hp[:]
        new_time = [wins_arg]  # time[:]
        # print("shape of result ", np.shape(new_pop))
        # print("shape of fit result ", np.shape(fitness), np.shape(new_fitness))
        return new_pop, new_fitness, new_php, new_ehp, new_time


    def crossover_mutate(env, pop, fitness, k, p_mutation, selection):
        n_pop = pop.shape[0]

        child_new = []
        if (selection == 'DE_old'):
            for i in range(n_pop):
                alpha = np.random.rand()
                while (True):
                    p1, p2 = np.random.randint(0, pop.shape[0], size=2)
                    if (p1 != i and p2 != i):
                        break
                x = pop[i]
                v = pop[p1] - pop[p2]
                u = x + alpha * v
                l = [x, u, v]

                f = evaluate(env, l)[:, 0]

                # print(np.shape(x), np.shape(v), np.shape(u), np.shape(f), f, np.shape(l[np.argmax(f)]))
                if f[0] >= f[1]:
                    if f[0] >= f[2]:
                        child = l[0]  # = np.vstack((child_new, l[0]))
                    else:
                        child = l[2]  # child_new = np.vstack((child_new, l[2]))
                else:
                    if f[1] >= f[2]:
                        child = l[1]  # child_new = np.vstack((child_new, l[1]))
                    else:
                        child = l[2]  # child_new = np.vstack((child_new, l[2]))

                if i == 0:
                    child_new = [child]
                else:
                    child_new = np.vstack((child_new, [child]))
                # print(np.shape(child_new))
        elif (selection == 'DE'):
            for i in range(n_pop):
                alpha = np.random.rand()
                while (True):
                    p1, p2 = np.random.randint(0, pop.shape[0], size=2)
                    if (p1 != i and p2 != i):
                        while (True):
                            #p3 = np.random.randint(0, pop.shape[0] / 2, size=1)
                            p3 = tournament(pop, fitness, k)
                            if (p1 != p3 and p2 != p3 and p3 != i):
                                break
                        if DEBUG_T == 1:
                            print("DE select basis vector:"+str(p1)+","+str(p2)+","+str(p3))
                        break

                x = pop[p3] #+ random_noise(n_vars, p_mutation)  # / 2)
                v = pop[p1] - pop[p2]
                #v = v + random_noise(n_vars, p_mutation)  # / 2)
                u = x + alpha * v
                #u = u + random_noise(n_vars, p_mutation)  # / 2)
                child1, child2 = cross_two_point(u, pop[i])
                child1 = np.array(child1)
                child2 = np.array(child2)
                child1 = 1 * child1 + random_noise(n_vars, p_mutation)  # / 2)
                child2 = 1 * child2 + random_noise(n_vars, p_mutation)  # / 2)
                #TODO mutate before or after crossover might different?
                if i == 0:
                    child_new = [child1,child2]
                else:
                    child_new = np.vstack((child_new, [child1,child2]))
                #print(np.shape(child_new))
        else:
            children = tournament_select(pop, k, fitness)
            i = 0

            while i < n_pop - i - 1:
                parent1, parent2 = children[i].astype(int), children[n_pop - i - 1].astype(int)
                i += 1
                # if (i == 0 and DEBUG_T == 1):
                # print("crossover parent: ", pop[parent1])
                child1, child2 = cross_two_point(pop[parent1] + random_noise(n_pop,p_mutation), pop[parent2])
                # if (i == 0 and DEBUG_T == 1):
                # print("crossover: ", child1, child2)
                child1 = np.array(child1)
                child2 = np.array(child2)
                child1 = 1 * child1 + random_noise(n_vars, p_mutation)  # / 2)
                child2 = 1 * child2 + random_noise(n_vars, p_mutation)  # / 2)
                l = [child1, child2]
                # print("shape1 ", np.shape(l))
                child_new += l
                # print("shape3 ", np.shape(child_new))
        return child_new

    def configure_results(pop, fitness, player_hp, enemy_hp, time, generation, ultimate_best):

        # mean
        data_fitness['mean'] = np.append(data_fitness['mean'], np.mean(fitness))
        data_player_hp['mean'] = np.append(data_player_hp['mean'], np.mean(player_hp))
        data_enemy_hp['mean'] = np.append(data_enemy_hp['mean'], np.mean(enemy_hp))
        data_time['mean'] = np.append(data_time['mean'], np.mean(time))

        # std
        data_fitness['std'] = np.append(data_fitness['std'], np.std(fitness))
        data_player_hp['std'] = np.append(data_player_hp['std'], np.std(player_hp))
        data_enemy_hp['std'] = np.append(data_enemy_hp['std'], np.std(enemy_hp))
        data_time['std'] = np.append(data_time['std'], np.std(time))

        # max
        data_fitness['max'] = np.append(data_fitness['max'], np.max(fitness))
        data_player_hp['max'] = np.append(data_player_hp['max'], np.max(player_hp))
        data_enemy_hp['max'] = np.append(data_enemy_hp['max'], np.max(enemy_hp))
        data_time['max'] = np.append(data_time['max'], np.max(time))

        index_best = np.argmax(fitness)
        pop_best = pop[index_best]
        fitness_best = fitness[index_best]

        print("  Min %s" % min(fitness))
        print("  Avg %s" % np.mean(fitness))
        print("  Std %s" % np.std(fitness))

        print("  best fitness: %s" % fitness_best)
        print("  player_hp: %s" % player_hp[index_best])
        print("  enemy_hp: %s" % enemy_hp[index_best])
        print("  time: %s " % time[index_best])

        if fitness_best > ultimate_best:
            print("ultimate best %s :" % index_best)
            ultimate_best = fitness_best
            np.savetxt(env.experiment_name + "/best.txt", pop_best)
        if fitness_best > winner["fitness"]:
            print("WINNER")
            winner["solution"] = pop_best
            winner["fitness"] = fitness_best

        genlist.append(generation)
        data_bestfit['fitness'] = np.append(data_bestfit['fitness'], fitness_best)
        data_bestfit['player_hp'] = np.append(data_bestfit['player_hp'], player_hp[index_best])
        data_bestfit['enemy_hp'] = np.append(data_bestfit['enemy_hp'], enemy_hp[index_best])
        data_bestfit['time'] = np.append(data_bestfit['time'], time[index_best])

        # save result of each generation
        # file_aux  = open(experiment_name+'/results.txt','a')
        # file_aux.write('\n\ngen best mean std')
        # file_aux.write('\n'+str(generation)+' '+str(round(max_fitness,6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        # file_aux.close()

        return ultimate_best

    def evolution(pop, ultimate_best, selection="default"):
        """
        Evolution Steps:
        1. Select next generation of individuals from population
        2. Clone is used (I think) to let the DEAP algorithm know it is a new generation
        3. Apply Crossover on the offspring
        4. Apply Mutation on the offspring
        5. Evaluate individuals that have been changed due to crossover or mutation
        6. Apply survivor selection by picking the best of a group
        6. Show statistics of the fitness levels of the population and save best individual of that run
        7. Update environment solutions

        Args:
            pop (nparray): A list containing individuals
            selection : "DE" | "default"
        """

        current_g = 0
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        while current_g < epoch:
            print("-- Generation %i --" % current_g)

            # 1. mate and crossover and/or mutate
            offspring = np.array(crossover_mutate(env, pop, fitness, n_pop, p_mutation, selection))
            # 2. evaluate offspring
            if (DEBUG_T == 1):
                pass #print(offspring.shape)
            results = evaluate(env, offspring)
            fitness = np.append(fitness, results[:, 0])
            player_hp = np.append(player_hp, results[:, 1])
            enemy_hp = np.append(enemy_hp, results[:, 2])
            time = np.append(time, results[:, 3])
            # if DEBUG_T == 1:
            # print(np.shape(pop), np.shape(offspring),np.shape(fitness),np.shape(player_hp),np.shape(enemy_hp),np.shape(time))
            pop = np.vstack((pop, offspring))
            # if DEBUG_T == 1:
            # print(np.shape(pop), np.shape(offspring), np.shape(fitness))

            # 3. survive select
            survivors, fitness_new, php_new, ehp_new, time_new = \
                round_robin_tournament_survivor(n_pop, pop, fitness, player_hp, enemy_hp, time)
            # shuffle_tournament_survivor(n_pop, pop, fitness, player_hp, enemy_hp, time)
            # select_best(n_pop, pop, fitness, player_hp, enemy_hp, time)
            # Replace old pop by selected pop
            pop = survivors[:]
            fitness = fitness_new[:]
            player_hp = php_new[:]
            enemy_hp = ehp_new[:]
            time = time_new[:]

            # 4. evaluate new pop
            ultimate_best = configure_results(pop, fitness, player_hp, enemy_hp, time, current_g, ultimate_best)

            # 8.
            solutions = [pop, fitness]
            env.update_solutions(solutions)
            env.save_state()
            current_g = current_g + 1
            if current_g % 100 == 0:
                save_pop(pop)
                print_2_csv(current_g)
        return pop
    def save_pop(pop):
        np.savetxt(env.experiment_name  + "/population.txt", pop)
    def print_2_csv(eponum = None, selection_tmp = None):
        print("SAVE RESULTS TO CSV")
        with open(env.experiment_name + '/results.csv', 'a+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            if eponum == 100:
                filewriter.writerow(
                    ["generation", "fitness_max", "mean", "std", "player_hp_max", "mean", "std", "enemy_hp_max", "mean",
                     "std", "time_max", "mean", "std"])

            for i in range(eponum-100,eponum):
                filewriter.writerow([i + 1, data_fitness['max'][i], data_fitness['mean'][i], data_fitness['std'][i],
                                     data_player_hp['max'][i], data_player_hp['mean'][i], data_player_hp['std'][i],
                                     data_enemy_hp['max'][i], data_enemy_hp['mean'][i], data_enemy_hp['std'][i],
                                     data_time['max'][i], data_time['mean'][i], data_time['std'][i],
                                     data_bestfit['fitness'][i], data_bestfit['player_hp'][i],
                                     data_bestfit['enemy_hp'][i], data_bestfit['time'][i]
                                     ])

    def coenvolve(pop1, pop2):
        pop = np.vstack((pop1, pop2))
        current_g = 0
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]

        # 3. survive select

        survivors, fitness_new, php_new, ehp_new, time_new = shuffle_tournament_survivor(n_pop, pop, fitness, player_hp,
                                                                                         enemy_hp, time)
        # survivors, fitness_new, php_new, ehp_new, time_new = select_best(n_pop, pop, fitness, player_hp,
        #                                                                enemy_hp, time)
        # Replace old pop by selected pop
        pop = survivors[:]
        return evolution(pop, -100, selection)

    def main(pop1, pop2):
        if mode == "all" and np.shape(pop2)[0] != 0 and np.shape(pop1)[0] != 0:
            pop = coenvolve(pop1, pop2)
        else:
            pop_ini = init_pop(n_pop, n_vars)
            pop = evolution(pop_ini, -100, selection)
        print_2_csv(epoch)
        return winner, pop

    if mode == "two":
        pop_ini = init_pop(n_pop, n_vars)
        pop_old = pop_ini[:]
        evolution(pop_ini, -100, selections[0])

        if not os.path.exists(experiment_name + "_" + selections[1]):
            os.makedirs(experiment_name + "_" + selections[1])

        if os.path.exists(experiment_name + "_" + selections[1] + '/results.csv'):
            os.remove(experiment_name + "_" + selections[1] + '/results.csv')

        if os.path.exists(experiment_name + "_" + selections[1] + '/best.txt'):
            os.remove(experiment_name + "_" + selections[1] + '/best.txt')

        env = Environment(
            experiment_name=experiment_name + "_" + selections[1],
            enemies=enemies_in_group,
            multiplemode="yes",
            playermode="ai",
            player_controller=player_controller(n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
        )
        # DATA
        genlist = []
        winner = {"solution": [], "fitness": -200}
        data_fitness = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
        data_player_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
        data_enemy_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
        data_time = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
        data_bestfit = {'fitness': np.array([]), 'player_hp': np.array([]), 'enemy_hp': np.array([]),
                        'time': np.array([])}

        evolution(pop_old, -100, selections[1])
    else:
        main(pop1, pop2)


def train(enemy_number, Continue, selection, index = 0):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
    experiment_name = 'solution/' + str(enemy_number)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name = experiment_name,
                    enemies = int2list(enemy_number),
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "fastest",
                    visuals = False,
                    multiplemode = 'yes')

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    n_pop = 100
    p_mutation = 0.2
    epoch = 100
    best_f = -1

    if(os.path.exists(experiment_name + '/pop_{}.bin'.format(index)) and Continue):
        pop = np.fromfile(experiment_name + '/pop_{}.bin'.format(index), dtype = np.float64).reshape(n_pop, n_vars)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        best_f = np.max(fitness)
        print('best_f {}'.format(best_f))
        sleep(1)
    else:
        pop = init(n_pop, n_vars)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        print("training from scratch")
        sleep(1)

    data_fitness = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_player_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_enemy_hp = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    data_time = {'mean': np.array([]), 'std': np.array([]), 'max': np.array([])}
    for i in range(epoch):
        # data collection
        # mean
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]  
        data_fitness['mean'] = np.append(data_fitness['mean'], np.mean(fitness))
        data_player_hp['mean'] = np.append(data_player_hp['mean'], np.mean(player_hp))
        data_enemy_hp['mean'] = np.append(data_enemy_hp['mean'], np.mean(enemy_hp))
        data_time['mean'] = np.append(data_time['mean'], np.mean(time))
        
        # std
        data_fitness['std'] = np.append(data_fitness['std'], np.std(fitness))
        data_player_hp['std'] = np.append(data_player_hp['std'], np.std(player_hp))
        data_enemy_hp['std'] = np.append(data_enemy_hp['std'], np.std(enemy_hp))
        data_time['std'] = np.append(data_time['std'], np.std(time))

        # max
        data_fitness['max'] = np.append(data_fitness['max'], np.max(fitness))
        data_player_hp['max'] = np.append(data_player_hp['max'], np.max(player_hp))
        data_enemy_hp['max'] = np.append(data_enemy_hp['max'], np.max(enemy_hp))
        data_time['max'] = np.append(data_time['max'], np.max(time))

        pop = crossover(env, pop, fitness, p_mutation, selection)
        results = evaluate(env, pop)
        fitness, player_hp, enemy_hp, time = results[:, 0], results[:, 1], results[:, 2], results[:, 3]        
        pop, fitness = select(n_pop, pop, fitness)

        index_best = np.argmax(fitness)
        pop_best = pop[index_best]
        fitness_best = fitness[index_best]

        # import pdb; pdb.set_trace()

        print('epoch {} best fitness {}'.format(i, fitness_best))

        if(fitness_best > best_f):
            print('best solution saved to {}/best_{}.bin and {}/pop_{}.bin'.format(experiment_name, index, experiment_name, index))
            pop_best.tofile(experiment_name + '/best_{}.bin'.format(index))
            pop.tofile(experiment_name + '/pop_{}.bin'.format(index))
            best_f = fitness_best
    
    return data_fitness, data_player_hp, data_enemy_hp, data_time


def test(enemy_number, index = 0, exp_name = None):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    n_hidden_neurons = 10
    if len(enemy_number) == 1:
        playmode='no'
    else:
        playmode='yes'

    if exp_name is not None:
        name = ''.join(str(x) for x in enemy_number)
        experiment_name = exp_name+"/test" #+ name
        print(experiment_name)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        env = Environment(experiment_name=experiment_name,
                          enemies=enemy_number,
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          # you can insert your own controller here
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=True,
                          multiplemode=playmode)
        # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
        pop = np.loadtxt(exp_name + '/best.txt').reshape(1, n_vars)
        #print(pop)
    else:
        experiment_name = 'solution/' + str(enemy_number)
        exp_name = 'solution/'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name = experiment_name,
                    enemies = int2list(enemy_number),
                    playermode = "ai",
                    player_controller = player_controller(n_hidden_neurons), # you can insert your own controller here
                    enemymode = "static",
                    level = 2,
                    speed = "normal",
                    visuals = False,
                    multiplemode = 'yes')
        # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

        pop = np.fromfile(exp_name + '/best.txt'.format(index), dtype=np.float64).reshape(1, n_vars)



    #results = evaluate_test(env, pop)[0]
    fitness, player_hp, enemy_hp, time = env.play(pcont=pop[0])
    #fitness, player_hp, enemy_hp, time = results[0], results[1], results[2], results[3]

    print('fitness {}, player_hp {}, enemy_hp {}, time {}'.format(fitness, player_hp, enemy_hp, time))

    return player_hp - enemy_hp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'generation_train')  # 'generation_train')  #
    parser.add_argument('-n', '--enemy_number', type = int, default = 1)
    parser.add_argument('-c', '--Continue', action = 'store_true')
    parser.add_argument('-s', '--seed', type = int, default = 0)
    parser.add_argument('--selection', type = str, default = 'random')
    #parser.add_argument('--noise', type=str, default='normal') # or uniform
    
    args = parser.parse_args()

    if(args.mode == 'train'):
        train(args.enemy_number, args.Continue, args.selection)
    elif(args.mode == 'test'):
        test(args.enemy_number)
    elif(args.mode == 'data'):
        experiment_name = 'solution/' + str(args.enemy_number)

        f = np.array([])
        p = np.array([])
        e = np.array([])
        t = np.array([])
        for i in range(10):
            data_f, data_p, data_e, data_t = train(args.enemy_number, False, args.selection, i)
            f = np.append(f, data_f)
            p = np.append(p, data_p)
            e = np.append(e, data_e)
            t = np.append(t, data_t)
        
        with open(experiment_name + '/data_f.pkl', 'wb') as file:
            pickle.dump(f, file)

    elif(args.mode == 'data_test'):
        data = {'score': []}
        for i in range(10):
            score = test(args.enemy_number, i)
            data['score'].append(score)
        
        experiment_name = 'solution/' + str(args.enemy_number)
        with open(experiment_name + '/data_score.pkl', 'wb') as file:
            pickle.dump(data, file)

    elif (args.mode == 'generation_test'):
        data = {'score': []}
        enemy_groups = {1: [1, 2, 3, 4, 5, 6, 7, 8]}  #, 2: [4, 6, 7], 3: [2, 6, 8]}
        solution_names = ['group12345678_p_0.2_DE',
                         'group12345678_p_0.2_random',
                         #'group2368_p_0.2_DE',
                         #'group2368_p_0.2_random',
                         #'group4567_p_0.2_DE',
                         #'group4567_p_0.2_random',
                         'solution']
        for exp_name in solution_names:
            for group_number, enemies_in_group in enemy_groups.items():
                #exp_name = 'group12345678_DE'
                print(enemies_in_group)
                score = test(enemies_in_group, group_number, exp_name)
                data['score'].append(score)
                """
                for i in range(8):
                    l = [enemies_in_group[i],enemies_in_group[7-i]]
                    print(l)
                    score = test(l, group_number, exp_name)
                    data['score'].append(score)
                """
                win = 0
                for i in range(8):
                    # exp_name = 'group12345678'
                    print(i)
                    score = test([i + 1], 1, exp_name)
                    if score > 0 :
                        win += 1
                    data['score'].append(score)
                print("solution "+exp_name+" Win "+str(win)+" enemies! ")


        print(data['score'])


    elif (args.mode == 'generation_train'):
        print("------------------------------- START TRAIN -------------------------------------------------------")
        # --------- STARTS PROGRAM FOR EVERY ENEMY GROUP 10 TIMES ---------------
        enemy_groups = {1: [1, 2, 3, 4, 5, 6, 7, 8], 2: [2, 3, 6, 8], 3: [4, 5, 6, 7], 4: [1, 2, 3, 4, 5, 6, 7, 8]}
        pops = [1, 2, 3]
        for group_number, enemies_in_group in enemy_groups.items():
            group_name = ''.join(str(x) for x in enemies_in_group)
            print("------------ GROUP " + str(group_name) + " -------------------------------------------------------")
            experiment_name = "exp_1_" + str(group_name)
            if group_number < 3:
                #continue
                # pops[group_number-1] =
                generalist_train(experiment_name, enemies_in_group, None, mode="two")
            elif group_number == 3:
                experiment_name = "group_coevo"
                # pops[group_number - 1] = generalist_train(experiment_name, enemies_in_group, args.selection, mode="all",
                # pop1=pops[0], pop2=pops[1])
            else:
                pass
                #generalist_train(experiment_name, enemies_in_group, None, mode="two")
