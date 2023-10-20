# Evoman

Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

## Structure
* initialization: uniform randomized 
  * normal initialization also provided
* parent selection and mate: 
  * EA 1: tournament: k = 2
  * EA 2: DE (differential evolution):
    * basis vector: tournament select
    * scale factor: random rand
* crossover: 2 point
* mutation: p = 0.2 (uniform)
* survivor selection: Round-robin (q = 10)
  * tournament select also provided

## Train
Run `optimization_dummy.py -m generation_train`
### Setting
hard-coded:
* n_pop = 100
* offspring size: 200
* epoch = 40
* keep_rate = 20 (not in use)
for calling:
* runs: number of times run
* selection: [DE|tournament]
* k: tournament num, used in both EAs
* p_mutation: 0.2
* mode=[None|two|all]
  * None: run one EA
  * two: run two EA with same initial population
  * all: co-evolution, run EA 1 and EA 2 sequentially, is not used, for bad performance
### results
For each experiment, the best solution is in best.txt, {fitness,hp and time}'s {mean, std and max} are saved to results.csv.

For runs > 1, {max, mean, std} of fitness of all runs are saved in results_all.csv

## Test
Run `optimization_dummy.py -m generation_test`

>Directory name in line with training output:

`test_group = [1, 2, 3, 4, 5, 6, 7, 8]`

`exp_groups = {1: 'exp_5_26', 2: 'exp_5_78'}`

`selection_names = ['_DE', '_tournament']`

experiment or solution directory name = {exp_groups} + {selection_names} combination

solution folder must contain {exp_runs} subdirectories (automat generate by setting training runs)
