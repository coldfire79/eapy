"""
read MATLAB file
"""

import numpy as np
import scipy.io
from glvmodel import GeneralizedLotkaVolterra


def read_mat(mat_file):
    mat = scipy.io.loadmat(mat_file)
    print(mat['model'].dtype.names)

    S = mat['model']['S'][0,0][0,0]
    r = mat['model']['r'][0, 0][0, 0] * np.ones(S)
    A = mat['model']['A'][0, 0]
    y = mat['model']['X'][0, 0]
    return S, r, A, y


if __name__ == "__main__":
    S, r, A, y = read_mat('/Volumes/HotLakeModeling/Network inference/Data for Processes paper/3_0.5_1_1.mat')
    print(S, list(r), A, y.shape)
    model = GeneralizedLotkaVolterra(S, np.array(list(r)), A)
    t = np.linspace(0, 50, num=5001)
    profile = model.generate_dynamic_profile(y[0, :], t)
    print(np.mean(np.abs(profile-y)))

    y_dot, A, R = model.fetch_matrices(profile, time_interval=0.01)
    print(y_dot, A, R)
    err = model.cost_fn(y_dot, A, R)
    print('err:', err)
