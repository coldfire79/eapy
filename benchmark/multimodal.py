"""
TEST functions
"""
from math import exp, sin, cos, sqrt


class BenchmarkMultiModal:
    def __init__(self, global_optima, dim, obj_fn, global_opt_val=None, epsilon=1.0E-3, r=1.0E-3):
        self.global_optima = global_optima
        self.dim = dim
        self.obj_fn = obj_fn
        self.epsilon = epsilon
        self.r = r
        if global_opt_val:
            self.global_opt_val = global_opt_val
        else:
            self.global_opt_val = self.obj_fn(self.global_optima)
        self.numFE = 0

    def eval(self, individual):
        if individual.is_evaluated:
            return individual.score
        else:
            self.numFE += 1
            individual.score = self.obj_fn(individual.vector)
            return individual.score

    def error(self, individual):
        return abs(self.global_opt_val - individual.score)
    # def is_solution(self, individual):
    #     rsm = sqrt(sum((x-y)**2 for x, y in zip(individual.vector, self.global_optima)))
    #     return (rsm <= self.epsilon) & ((self.global_opt_val - individual.score) <= self.r)


class Shubert(BenchmarkMultiModal):
    """Shubert test objective function.
        minimization version of benchmark function
            in Benchmark Functions for CEC 2015 Special Session and Competition on Dynamic
    """
    def __init__(self, dim, epsilon=1.0E-3, r=1.0E-3):
        # set global optimum
        global_optima = [1] * dim
        global_opt_val = -186.73

        def fn(individual):
            ans = 1
            for x in individual:
                ans *= sum(j*cos((j+1)*x+j) for j in range(1, 6))
            return ans

        super().__init__(global_optima, dim, fn, global_opt_val=global_opt_val, epsilon=epsilon, r=r)
