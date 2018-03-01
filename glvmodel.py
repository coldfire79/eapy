"""
Lotka-Volterra equations for n species.

    xdots_i = x_i * (r_i + sum_j(a_ij * x_j))

x_i denotes the density of the i-th species, r_i is its intrinsic
growth (or decay) rate and the matrix A = (a_ij) is called the interaction
matrix.
"""
import numpy as np
from scipy.integrate import ode

class GeneralizedLotkaVolterra:
    def __init__(self, n_species=3, growth_rate=None, interaction_matrix=None, dt=0.01):
        self.n_species = n_species
        self.growth_rate = np.ones(n_species)
        self.interaction_matrix = np.identity(n_species)
        self.dt = dt
        if growth_rate is not None:
            self.growth_rate = growth_rate
        if interaction_matrix is not None:
            self.interaction_matrix = interaction_matrix

    def glv(self, t, x, r, a):
        n = self.n_species
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = x[i] * (r[i] + np.dot(a[i, :], x))
        return dx

    def generate_dynamic_profile(self, x0, t):  # (self, r, A, x0, t):
        print('growth rate', self.growth_rate)
        print('A', self.interaction_matrix)
        ode_obj = ode(self.glv).set_integrator('dop853', nsteps=30)
        ode_obj.set_initial_value(x0, 0).set_f_params(self.growth_rate, self.interaction_matrix)

        t1 = t[-1]
        dt = self.dt
        sol = np.zeros((len(t), self.n_species))
        i = 0
        sol[i, :] = x0

        while ode_obj.successful() and i + 1 < len(t):
            i += 1
            ode_obj.integrate(t[i])
            sol[i, :] = ode_obj.y
        return sol

    # def show_profile(self, x0):


if __name__ == "__main__":
    test()