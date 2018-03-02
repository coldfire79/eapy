"""
BNDE: Bare-Bones Niching DE
"""

import random
import numpy as np
import scipy.io
import initialize as init
from benchmark.multimodal import BenchmarkMultiModal
from population import Individual, MultiPopulationsWithSameSize
from glvmodel import GeneralizedLotkaVolterra


def gaussian_mutation(x_best, x_target, bounds):
    _mean = [(x_best_i + x_target_i)/2 for x_best_i, x_target_i in zip(x_best.vector, x_target.vector)]
    _std = [abs(x_best_i - x_target_i) for x_best_i, x_target_i in zip(x_best.vector, x_target.vector)]
    v_donor = [np.random.normal(mu, sigma) for mu, sigma in zip(_mean, _std)]
    v_donor = ensure_bounds(v_donor, bounds)
    return v_donor


def gaussian_mutation_bnde(best_idx, target_idx, population, bounds, PE, chi):
    pop_size = population.size
    # select three random vector index positions [0, pop_size), not including current vector (j)
    candidates = [k for k in range(pop_size) if (k != target_idx) & (k != best_idx)]
    random_index = random.sample(candidates, 1)
    x_rand = population.get(random_index[0]).vector
    x_best =  population.get(best_idx).vector
    x_t = population.get(target_idx).vector

    _mean = x_best

    prob = np.random.rand()
    if prob > PE:
        _std = [abs(x_rand_i - x_target_i) for x_rand_i, x_target_i in zip(x_rand, x_t)]
    else:
        _std = [chi * (b[1]-b[0]) for b in bounds]
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
        # print(score_trial, "<=", score_target)
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


def euclidean_dist(list1, list2):
    return np.linalg.norm(np.array(list1)-np.array(list2))


def diversity_preserving(neighbors, archive, bounds, cost_fn, overlapping_threshold=0.01, d0=1.0E-16):
    S = []
    centers = neighbors.get_centers()
    for i in range(neighbors.size):
        sub_pop_i = neighbors.get_subpopulation(i)
        best_i, worst_i = sub_pop_i.get_best_worst(redo=False)
        if i in S:
            continue
        # the neighborhood is considered as converged once
        random_index = np.random.randint(sub_pop_i.size)
        r_i = euclidean_dist(centers[i], sub_pop_i.get(random_index).vector)
        if r_i <= d0:
            # print("Neighborhood converged:{0}".format(str(best_i)))
            archive.append(best_i)
            new_pop = reinitialize_population(bounds, cost_fn, sub_pop_i)
            neighbors.update_subpopulation(i, new_pop)
            # print("{0}: reinitialized".format(i))
            # print("new_pop")
            # print(str(new_pop))
            # print("neighbors")
            # print(str(neighbors))
            S.append(i)
        for j in range(i+1, neighbors.size):
            d_i_j = euclidean_dist(centers[i], centers[j])
            if (j in S) | (d_i_j > overlapping_threshold):
                continue
            sub_pop_j = neighbors.get_subpopulation(j)
            best_j, worst_j = sub_pop_j.get_best_worst(redo=False)
            # for minimization problem: TODO
            if best_i.score <= best_j.score:
                if best_j.score <= worst_i.score:
                    # print(neighbors.get_subpopulation_individual(i, sub_pop_i.worst_index))
                    # print(best_j)
                    neighbors.update_individual(i, sub_pop_i.worst_index, best_j)
                    # print("neighbors.update_individual(i, sub_pop_i.worst_index, best_j)")
                    # print(neighbors.get_subpopulation_individual(i, sub_pop_i.worst_index))
                new_pop = reinitialize_population(bounds, cost_fn, sub_pop_j)
                neighbors.update_subpopulation(j, new_pop)
                # print("{0}: reinitialized".format(j))
                # print("new_pop")
                # print(str(new_pop))
                # print("neighbors")
                # print(str(neighbors))
                S.append(j)
            else:
                if best_i.score <= worst_j.score:
                    # print(neighbors.get_subpopulation_individual(j, sub_pop_j.worst_index))
                    # print(best_i)
                    neighbors.update_individual(j, sub_pop_j.worst_index, best_i)
                    # print("neighbors.update_individual(j, sub_pop_j.worst_index, best_i)")
                    # print(neighbors.get_subpopulation_individual(j, sub_pop_j.worst_index))
                new_pop = reinitialize_population(bounds, cost_fn, sub_pop_i)
                neighbors.update_subpopulation(i, new_pop)
                # print("{0}: reinitialized".format(i))
                # print("new_pop")
                # print(str(new_pop))
                # print("neighbors")
                # print(str(neighbors))
                S.append(i)
                break
    return neighbors


def reinitialize_population(bounds, cost_fn, sub_pop):
    new_pop = init.uniform_random(sub_pop.size, bounds)
    init_eval(new_pop, cost_fn)
    return new_pop


def update_pe(num_neighbors, neighborhoodSize, mu_pe, q=0.1):
    next_pe = []
    pop_total = num_neighbors * neighborhoodSize
    _pe = np.random.normal(mu_pe, 0.1, pop_total)
    _pe[_pe < 0] = 0
    _pe[_pe > 1] = 1
    for i in range(num_neighbors):
        next_pe.append(list(_pe[i*neighborhoodSize:(i+1)*neighborhoodSize]))
    return next_pe, (1-q)*mu_pe + q*np.mean(next_pe)


def update_cr_bnde(num_neighbors, neighborhoodSize, mu_cr, q=0.1):
    next_cr = []
    pop_total = num_neighbors * neighborhoodSize
    _tmp = np.random.normal(mu_cr, 0.1, pop_total)
    _tmp[_tmp < 0] = 0
    _tmp[_tmp > 1] = 1
    for i in range(num_neighbors):
        next_cr.append(list(_tmp[i*neighborhoodSize:(i+1)*neighborhoodSize]))
    return next_cr, (1 - q) * mu_cr + q * np.mean(next_cr)

def bnde_run(benchmark, bounds, pop_size, neighborhoodSize, maxFE, d0=1.0E-16):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    archive = []  # a archive to store all the local best

    population = init.uniform_random(pop_size, bounds)
    print(str(population))
    best = init_eval(population, benchmark.eval)
    print(str(population))
    neighbors = MultiPopulationsWithSameSize(pop_size, neighborhoodSize, population)
    print(str(neighbors))


    # cr = init_cr(pop_size)
    # mutation_strategy = init_mutation_strategies(pop_size)

    mu_pe = 0.5
    mu_cr = 0.5

    #--- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    while benchmark.numFE < maxFE:
        # best = init_eval(population, benchmark.eval)
        pe, mu_pe = update_pe(neighbors.size, neighborhoodSize, mu_pe, q=0.1)
        cr, mu_cr = update_cr_bnde(neighbors.size, neighborhoodSize, mu_cr, q=0.1)

        diversity_preserving(neighbors, archive, bounds, benchmark.eval, d0=d0)
        chi = np.exp(-4.0*(benchmark.numFE/maxFE+0.4))
        # cycle through each individual in the population
        best_scores = []
        for i in range(neighbors.size):
            sub_pop_i = neighbors.get_subpopulation(i)
            best_i, worst_i = sub_pop_i.get_best_worst(redo=True)
            best_scores.append(benchmark.error(best_i))
            print(best_i)
            for j in range(sub_pop_i.size):
                x_t = sub_pop_i.get(j).vector

                #--- MUTATION (step #3.A) ---------------------+

                v_donor = gaussian_mutation_bnde(sub_pop_i.best_index, j, sub_pop_i, bounds, pe[i][j], chi)
                # print("v_donor", v_donor)
                # if mutation_strategy[j] == 0:
                #     v_donor = gaussian_mutation(best, population.get(j), bounds)
                # else:
                #     v_donor = de_best_1_mutation(best, population, j, bounds, F)

                #--- RECOMBINATION (step #3.B) ----------------+
                v_trial = crossover(v_donor, x_t, cr[i][j])
                # print("v_trial", v_trial)

                #--- GREEDY SELECTION (step #3.C) -------------+
                score_trial, score_target = selection(sub_pop_i, j, benchmark.eval, Individual(v_trial))
                # print("numFE:", benchmark.numFE)
        #--- SCORE KEEPING --------------------------------+

        # gen_avg = sum(gen_scores) / pop_size                         # current generation avg. fitness
        # gen_best = min(gen_scores)                                  # fitness of best individual
        # gen_sol = population.get(gen_scores.index(min(gen_scores)))     # solution of best individual

        if benchmark.numFE % 1000 == 0:
            print('numFE:', benchmark.numFE)
            # print('      > GENERATION AVERAGE:', gen_avg)
            # print('      > GENERATION BEST:', gen_best)
            # print('         > BEST SOLUTION:', gen_sol)
            print('----------------------------------------------')
            print('      > GENERATION AVG BEST:', np.mean(best_scores))
            print('      > GENERATION STD BEST:', np.std(best_scores))
            print('      > GENERATION BEST of BEST:', np.min(best_scores))
            print('      > GENERATION WORST of BEST:', np.max(best_scores))
            print('----------------------------------------------')
    for solution in archive:
        print(solution)
    return population


class LVInference(BenchmarkMultiModal):
    def __init__(self, dim, num_species, growth_rate, y_dot, A, R, true_interactions, epsilon=1.0E-3, r=1.0E-3):
        global_opt_val = 0

        half = dim//2

        # set global optimum
        global_optima = [1] * half + list(true_interactions.flatten())

        def fn(individual):
            # nonzeros = [0 if i < 0.5 else 1 for i in individual[0:half]]
            # nonzeros[nonzeros < 0.5] = 0
            # nonzeros[nonzeros >= 0.5] = 1

            # interactions = [m * n for m, n in zip(nonzeros, individual[half:])]
            interactions = individual[half:]

            model = GeneralizedLotkaVolterra(num_species, growth_rate, np.array(interactions).reshape(-1, num_species))
            err = model.cost_fn(y_dot, A, R)
            return err

        super().__init__(global_optima, dim, fn, global_opt_val=global_opt_val, epsilon=epsilon, r=r)


def read_mat(mat_file):
    mat = scipy.io.loadmat(mat_file)
    print(mat['model'].dtype.names)

    S = mat['model']['S'][0,0][0,0]
    r = mat['model']['r'][0, 0][0, 0] * np.ones(S)
    A = mat['model']['A'][0, 0]
    y = mat['model']['X'][0, 0]
    return S, r, A, y


def test():
    random.seed(0)

    ################ reference model
    num_species, r_ans, A_ans, y_ans = read_mat('/Volumes/HotLakeModeling/Network inference/Data for Processes paper/3_0.5_1_1.mat')
    print("Num of Species:{0}".format(num_species))
    print("Growth rate:{0}".format(list(r_ans)))
    print("Interaction matrix:{0}".format(A_ans))
    t = np.linspace(0, 50, num=5001)

    model = GeneralizedLotkaVolterra(num_species, np.array(list(r_ans)), A_ans)
    profile = model.generate_dynamic_profile(y_ans[0, :], t)
    print(np.mean(np.abs(profile - y_ans)))

    y_dot, A, R = model.fetch_matrices(profile, time_interval=0.01)
    print(y_dot, A, R)
    err = model.cost_fn(y_dot, A, R)
    print("optimum err:", err)
    ################

    dim = 2 * num_species * num_species
    benchmark = LVInference(dim, num_species, np.array(list(r_ans)), y_dot, A, R, A_ans, r=err)
    bounds = [[0, 1]]*dim
    pop_size = 200
    F = 0.5
    CR = 0.9
    maxFE = 20000

    neighborhoodSize = 5
    numNeighborhoods = pop_size // neighborhoodSize

    d0 = 10 ** (-16/np.sqrt(dim))
    # --- RUN ----------------------------------------------------------------------+

    bnde_run(benchmark, bounds, pop_size, neighborhoodSize, maxFE, d0)


if __name__ == "__main__":
    test()
