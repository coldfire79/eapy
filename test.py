"""
read MATLAB file
"""

import numpy as np
import scipy.io
from glvmodel import GeneralizedLotkaVolterra

def read_mat(mat_file):
    mat = scipy.io.loadmat(mat_file)
    print(mat['model'].dtype.names)
    r = mat['model']['r'][0, 0].flatten()
    S = mat['model']['S'][0, 0].flatten()
    A = mat['model']['A'][0, 0]
    x = mat['model']['X'][0, 0]
    return S, r, A, x


if __name__ == "__main__":
    S, r, A, x = read_mat('/Volumes/HotLakeModeling/Network inference/Data for Processes paper/3_0.5_1_1.mat')
    print(S, list(r)*3, A, x.shape)
    model = GeneralizedLotkaVolterra(S[0], np.array(list(r)*S[0]), A)
    t = np.linspace(0, 50, num=5001)
    profile = model.generate_dynamic_profile(x[0,:], t)
    print(np.mean(np.abs(profile-x)))