"""
GBDE: Gaussian bare-bones differential evolution
"""

import random
import numpy as np
import initialize as init
from benchmark.common import Rosenbrock, Rosenbrock2
from population import Individual


def gaussian_mutation(x_best, x_target, bounds):
    _mean = [(x_best_i + x_target_i)/2 for x_best_i, x_target_i in zip(x_best.vector, x_target.vector)]
    _std = [abs(x_best_i - x_target_i) for x_best_i, x_target_i in zip(x_best.vector, x_target.vector)]
    v_donor = [np.random.normal(mu, sigma) for mu, sigma in zip(_mean, _std)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def de_best_1_mutation(x_best, population, target_idx, bounds, F):
    pop_size = population.size
    # select three random vector index positions [0, pop_size), not including current vector (j)
    candidates = [k for k in range(pop_size) if k != target_idx]
    random_index = random.sample(candidates, 2)
    x_1 = population.get(random_index[0]).vector
    x_2 = population.get(random_index[1]).vector

    # subtract x3 from x2, and create a new vector (x_diff)
    x_diff = [x_1_i - x_2_i for x_1_i, x_2_i in zip(x_1, x_2)]
    # multiply x_diff by the mutation factor (F) and add to x_1
    v_donor = [x_1_i + F * x_diff_i for x_1_i, x_diff_i in zip(x_best.vector, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def mutation(population, target_idx, bounds, F):
    pop_size = population.size
    # select three random vector index positions [0, pop_size), not including current vector (j)
    candidates = [k for k in range(pop_size) if k != target_idx]
    random_index = random.sample(candidates, 3)
    x_1 = population.get(random_index[0]).vector
    x_2 = population.get(random_index[1]).vector
    x_3 = population.get(random_index[2]).vector

    # subtract x3 from x2, and create a new vector (x_diff)
    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]
    # multiply x_diff by the mutation factor (F) and add to x_1
    v_donor = [x_1_i + F * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def crossover(v_donor, v_target, CR):
    v_trial = []
    for k in range(len(v_target)):
        p = random.random()
        if p <= CR:
            v_trial.append(v_donor[k])
        else:
            v_trial.append(v_target[k])
    return v_trial


def selection(population, target_idx, cost_fn, v_trial_individual):
    score_trial = cost_fn(v_trial_individual)
    score_target = cost_fn(population.get(target_idx))
    if score_trial <= score_target:
        population.set(target_idx, v_trial_individual)
    return score_trial, score_target


def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):

        # variable exceeds the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceeds the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])

    return vec_new


def init_eval(population, cost_fn):
    for i in range(population.size):
        cost_fn(population.get(i))
    return population.get_best()


def init_cr(pop_size):
    return list(np.random.normal(0.5, 0.1, pop_size))


def init_mutation_strategies(pop_size):
    prob = np.random.rand(pop_size)
    return [1 if i > 0.5 else 0 for i in prob]


def update_cr(score_trial, score_target, cr):
    if score_trial <= score_target:
        return cr
    else:
        return np.random.normal(0.5, 0.1)


def gbde_run(benchmark, bounds, pop_size, maxFE):

    #--- INITIALIZE A POPULATION (step #1) ----------------+

    population = init.uniform_random(pop_size, bounds)
    cr = init_cr(pop_size)

    #--- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    while benchmark.numFE < maxFE:
        best = init_eval(population, benchmark.eval)
        # cycle through each individual in the population
        for j in range(pop_size):
            x_t = population.get(j).vector

            #--- MUTATION (step #3.A) ---------------------+

            v_donor = gaussian_mutation(best, population.get(j), bounds)

            #--- RECOMBINATION (step #3.B) ----------------+
            v_trial = crossover(v_donor, x_t, cr[j])

            #--- GREEDY SELECTION (step #3.C) -------------+
            score_trial, score_target = selection(population, j, benchmark.eval, Individual(v_trial))

            # --- UPDATE CR (step #3.C) -------------+
            cr[j] = update_cr(score_trial, score_target, cr[j])

        #--- SCORE KEEPING --------------------------------+

        # gen_avg = sum(gen_scores) / pop_size                         # current generation avg. fitness
        # gen_best = min(gen_scores)                                  # fitness of best individual
        # gen_sol = population.get(gen_scores.index(min(gen_scores)))     # solution of best individual
        best = population.get_best()
        if benchmark.numFE % 10000 == 0:
            print('numFE:', benchmark.numFE)
            # print('      > GENERATION AVERAGE:', gen_avg)
            # print('      > GENERATION BEST:', gen_best)
            # print('         > BEST SOLUTION:', gen_sol)
            print('----------------------------------------------')
            print('      > GENERATION BEST:', best.score)
            print('----------------------------------------------')
        if benchmark.is_solution(best):
            print('Find a solution! numFE:{0}'.format(benchmark.numFE))
            break
    return population


def mgbde_run(benchmark, bounds, pop_size, maxFE, F):

    #--- INITIALIZE A POPULATION (step #1) ----------------+

    population = init.uniform_random(pop_size, bounds)
    cr = init_cr(pop_size)
    mutation_strategy = init_mutation_strategies(pop_size)

    #--- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    while benchmark.numFE < maxFE:
        best = init_eval(population, benchmark.eval)
        # cycle through each individual in the population
        for j in range(pop_size):
            x_t = population.get(j).vector

            #--- MUTATION (step #3.A) ---------------------+
            if mutation_strategy[j] == 0:
                v_donor = gaussian_mutation(best, population.get(j), bounds)
            else:
                v_donor = de_best_1_mutation(best, population, j, bounds, F)

            #--- RECOMBINATION (step #3.B) ----------------+
            v_trial = crossover(v_donor, x_t, cr[j])

            #--- GREEDY SELECTION (step #3.C) -------------+
            score_trial, score_target = selection(population, j, benchmark.eval, Individual(v_trial))

            # --- UPDATE CR (step #3.C) -------------+
            cr[j] = update_cr(score_trial, score_target, cr[j])

        #--- SCORE KEEPING --------------------------------+

        # gen_avg = sum(gen_scores) / pop_size                         # current generation avg. fitness
        # gen_best = min(gen_scores)                                  # fitness of best individual
        # gen_sol = population.get(gen_scores.index(min(gen_scores)))     # solution of best individual
        best = population.get_best()
        if benchmark.numFE % 10000 == 0:
            print('numFE:', benchmark.numFE)
            # print('      > GENERATION AVERAGE:', gen_avg)
            # print('      > GENERATION BEST:', gen_best)
            # print('         > BEST SOLUTION:', gen_sol)
            print('----------------------------------------------')
            print('      > GENERATION BEST:', best.score)
            print('      > GENERATION BEST:', best.vector)
            print('----------------------------------------------')
        if benchmark.is_solution(best):
            print('Find a solution! numFE:{0}'.format(benchmark.numFE))
            break
    return population


def test():
    # random.seed(0)
    dim = 30
    benchmark = Rosenbrock2(dim)
    bounds = [[-30, 30]]*dim
    pop_size = 100
    F = 0.5
    CR = 0.9
    maxFE = 800000

    # --- RUN ----------------------------------------------------------------------+

    # gbde_run(benchmark, bounds, pop_size, maxFE)
    mgbde_run(benchmark, bounds, pop_size, maxFE, F)


if __name__ == "__main__":
    test()
