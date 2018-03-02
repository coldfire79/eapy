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

    def generate_dynamic_profile(self, x0, t):
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

    def fetch_matrices(self, dyn_profiles, time_interval=0.05):
        time_steps, num_species = dyn_profiles.shape
        # generate y
        y_dot = np.gradient(dyn_profiles, time_interval, axis=0).flatten()

        nm = num_species * time_steps
        # A
        A = np.zeros((nm, num_species * num_species))
        for i in range(time_steps):
            for j in range(num_species):
                A[num_species * i + j, num_species * j:num_species * (j + 1)] = dyn_profiles[i, :] * dyn_profiles[i, j]
        # R
        R = np.zeros((nm, num_species))
        for i in range(time_steps):
            R[num_species * i:num_species * (i + 1), :] = np.diag(dyn_profiles[i, :])
        return y_dot, A, R

    def cost_fn(self, y_dot, A, R):
        _y_dot = y_dot.reshape(-1, 1)
        # C = [A R]
        C = np.concatenate((A, R), axis=1)
        # print(C.shape)
        # b = [a; r]
        # print(self.interaction_matrix, self.growth_rate)
        b = np.concatenate((self.interaction_matrix.flatten(), self.growth_rate), axis=0).reshape(-1, 1)

        # H = np.matmul(C.T, C)
        # f = np.matmul(y.T, C)
        # term1 = np.matmul(np.matmul(b.T, H), b) / 2
        # term2 = np.matmul(f, b)
        # term3 = np.matmul(y.T, y) / 2
        #
        # out = y - np.matmul(C, b)
        #
        # print(term1 - term2 + term3)
        l2 = np.linalg.norm(_y_dot - np.matmul(C, b))
        # print(l2 * l2 / 2)
        return l2 * l2 / 2
