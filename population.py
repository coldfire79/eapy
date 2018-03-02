"""
Population: a set of individuals
"""
import sys
import numpy as np


class Individual:
    def __init__(self, vector):
        self._vector = vector
        self._score = -1
        self._is_evaluated = False

    def __str__(self):
        return "{0}\t{1}\t{2}".format(self._vector, self._score, self._is_evaluated)

    @property
    def vector(self):
        return self._vector

    @property
    def score(self):
        return self._score

    @property
    def is_evaluated(self):
        return self._is_evaluated

    @vector.setter
    def vector(self, vector):
        self._vector = vector

    @score.setter
    def score(self, score):
        self.is_evaluated = True
        self._score = score

    @is_evaluated.setter
    def is_evaluated(self, is_evaluated):
        self._is_evaluated = is_evaluated

    def get(self):
        return self.vector

    def update(self, vector):
        self.vector = vector
        self.score = -1
        self.is_evaluated = False


class Population:
    def __init__(self, size, mode="minimization"):
        self._size = size
        self._individuals = []
        self._scores = []

        self._mode = mode
        self._best_score = self.init_best_score()
        self._worst_score = self.init_best_score()

        self._best_index = -1
        self._worst_index = -1

        self._best_individual = None
        self._worst_individual = None

    # @property
    # def individuals(self):
    #     return self._individuals

    # @property
    # def score(self):
    #     return self._score

    def __str__(self):
        _str = ""
        for elem in self._individuals:
            _str += str(elem) + "\n"
        return _str

    @property
    def size(self):
        return self._size

    @property
    def best_index(self):
        return self._best_index

    @property
    def worst_index(self):
        return self._worst_index

    def init_best_score(self):
        if self._mode == "minimization":
            best_score = float(sys.maxsize)
        else:
            best_score = -float(sys.maxsize)
        return best_score

    def argminmax(self, arr):
        _max = -float(sys.maxsize)
        _min = float(sys.maxsize)
        max_idx = 0
        min_idx = 0
        for i, a in enumerate(arr):
            if a > _max:
                _max = a
                max_idx = i
            if a < _min:
                _min = a
                min_idx = i
        return min_idx, max_idx

    def update_best_score(self, scores):
        min_idx, max_idx = self.argminmax(scores)
        if self._mode == "minimization":
            self._best_score = scores[min_idx]
            self._best_individual = self._individuals[min_idx]
            self._worst_score = scores[max_idx]
            self._worst_individual = self._individuals[max_idx]
            self._best_index = min_idx
            self._worst_index = max_idx
        else:
            self._best_score = scores[max_idx]
            self._best_individual = self._individuals[max_idx]
            self._worst_score = scores[min_idx]
            self._worst_individual = self._individuals[min_idx]
            self._best_index = max_idx
            self._worst_index = min_idx

    def clear(self):
        self._individuals = []
        self._scores = []
        self._best_individual.update([])

    def add_individual(self, individual):
        assert type(individual) is not "Individaul", \
            "You can add only Individual-typed object."
        self._individuals.append(individual)

    def get(self, i):
        return self._individuals[i]

    def set(self, i, new_individual):
        self._individuals[i] = new_individual

    def get_scores(self):
        scores = []
        for individual in self._individuals:
            scores.append(individual.score)
        return scores

    def get_best(self, redo=True):
        if redo:
            self._scores = self.get_scores()
            self.update_best_score(self._scores)
        else:
            return self._best_individual

    def get_best_worst(self, redo=True):
        if redo | (self._best_individual is None):
            self._scores = self.get_scores()
            self.update_best_score(self._scores)
        return self._best_individual, self._worst_individual

    def get_center(self):
        s = len(self._individuals)
        if s > 0:
            center = self._individuals[0].vector
            for i in range(1, s):
                center = [x + y for x, y in zip(center, self._individuals[i].vector)]
            return [x/s for x in center]
        return None

    def replace(self, new_pop):
        for i in range(self._size):
            self._individuals[i] = new_pop.get(i)
        self._best_score = self.init_best_score()
        self._best_individual = None

    def copy_individual(self, i, new_individual):
        self._individuals[i].vector = new_individual.vector
        self._individuals[i].score = new_individual.score


class MultiPopulationsWithSameSize:
    def __init__(self, total_size, subpopulation_size, population, mode="minimization"):
        self._subpopulation_size = subpopulation_size
        self._total_size = total_size
        self._num_subpopulations = total_size // subpopulation_size
        print("Total Population:", self._total_size)
        print("Num. SubPopulations:", self._num_subpopulations)
        print("Num. Individuals per SubPopulation:", self._subpopulation_size)
        assert (self._num_subpopulations * subpopulation_size) == total_size

        self._subpopulations = []

        for i in range(self._num_subpopulations):
            neighborhood = Population(subpopulation_size, mode)
            for j in range(self._subpopulation_size):
                individual = population.get(i * self._subpopulation_size + j)
                neighborhood.add_individual(individual)
            self._subpopulations.append(neighborhood)

    def __str__(self):
        _str = ""
        for elem in self._subpopulations:
            _str += str(elem) + "\n"
        return _str

    @property
    def size(self):
        return self._num_subpopulations

    def get_subpopulation(self, i):
        return self._subpopulations[i]

    def get_subpopulation_individual(self, i, j):
        return self._subpopulations[i].get(j)

    def get_centers(self):
        centers = []
        for sub_pop in self._subpopulations:
            centers.append(sub_pop.get_center())
        return centers

    def update_subpopulation(self, i, new_subpopulation):
        self._subpopulations[i] = new_subpopulation

    def update_individual(self, i, j, new_individual):
        self._subpopulations[i].copy_individual(j, new_individual)