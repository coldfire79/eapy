"""
for initialization
"""
import random
from population import Population, Individual


def uniform_random(pop_size, bounds, mode="minimization"):
    population = Population(pop_size, mode)
    for i in range(pop_size):
        vector = []
        for j in range(len(bounds)):
            vector.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.add_individual(Individual(vector))
    return population

