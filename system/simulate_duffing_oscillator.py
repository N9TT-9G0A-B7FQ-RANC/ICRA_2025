import sys
sys.path.append('./')

from system import duffing_oscillator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import create_folder_if_not_exists

def noise(idx):
    if idx == 0:
        return np.random.normal(0, 0.15)
    if idx == 1:
        return np.random.normal(0, 0.3)
    if idx == 2:
        return np.random.normal(0, 0.45)

if __name__ == '__main__':

    folder_letter = ['a', 'b', 'c']
    
    delta_t = 0.001
    T = 20
    nb_integration_step = int(T / delta_t)
    nb_simulation = 200
    seed = 42

    folder_letter = ['a', 'b', 'c']

    intial_conditions = (np.random.rand(nb_simulation, 2) - 0.5) *  1

    noise_std = [0.15, 0.3, 0.45]
    var_name = ['x_', 'y_']
    
    for idx in range(2,3):

        folder_path = f'./simulation/duffing_simulation/'
        create_folder_if_not_exists(folder_path)

        for n in range(nb_simulation):

            t = 0

            results = {'x' : [], 'y':[], 'x_' : [], 'y_':[], 'f':[]}

            X = intial_conditions[n]
            
            for i in range(nb_integration_step):
                
                f = 0.3 * np.cos(1*t)

                results['x'].append(X[0] + noise(idx))
                results['y'].append(X[1] + noise(idx))
                
                results['f'].append(f)
                results['x_'].append(X[0])
                results['y_'].append(X[1])
                Xdt = duffing_oscillator(X, f)
                X = np.asarray(Xdt) * delta_t + X
                t += delta_t

            data = pd.DataFrame(results)
            data.to_csv(folder_path+f'simulation_{n}.csv')