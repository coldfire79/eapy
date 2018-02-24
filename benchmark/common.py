"""
TEST functions
"""
from math import exp, sin, cos, sqrt


class Benchmark:
    def __init__(self, global_optima, dim, obj_fn, epsilon=1.0E-3, r=1.0E-3):
        self.global_optima = global_optima
        self.dim = dim
        self.obj_fn = obj_fn
        self.epsilon = epsilon
        self.r = r
        self.global_opt_val = self.obj_fn(self.global_optima)
        self.numFE = 0

    def eval(self, individual):
        if individual.is_evaluated:
            return individual.score
        else:
            self.numFE += 1
            individual.score = self.obj_fn(individual.vector)
            individual.is_evaluated = True
            return individual.score

    def is_solution(self, individual):
        rsm = sqrt(sum((x-y)**2 for x, y in zip(individual.vector, self.global_optima)))
        return (rsm <= self.epsilon) & ((self.global_opt_val - individual.score) <= self.r)


class Rosenbrock(Benchmark):
    """Rosenbrock test objective function.
        benchmark function in DEAP
    """
    def __init__(self, dim, epsilon=1.0E-3, r=1.0E-3):
        # set global optimum
        global_optima = [1] * dim

        def fn(individual):
            return sum(100 * (x * x - y) ** 2 + (1. - x) ** 2 for x, y in zip(individual[:-1], individual[1:]))

        super().__init__(global_optima, dim, fn, epsilon, r)


class Rosenbrock2(Benchmark):
    """Rosenbrock test objective function.
        f5 in Gaussian bare-bones differential evolution
    """
    def __init__(self, dim, epsilon=1.0E-3, r=1.0E-3):
        # set global optimum
        global_optima = [1] * dim

        def fn(individual):
            return sum(100 * (y - x * x) ** 2 + (1. - x * x) ** 2 for x, y in zip(individual[:-1], individual[1:]))

        super().__init__(global_optima, dim, fn, epsilon, r)