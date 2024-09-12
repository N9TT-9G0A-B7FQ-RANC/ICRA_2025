import sys
sys.path.append('./')
from system import DOF10SimplifiedPacejka, DOF10SimplifiedPacejka_parameters
import vehicle_parameters
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import create_folder_if_not_exists
from tqdm import tqdm

if __name__ == '__main__':
    
    delta_t = 0.001
    T = 20
    nb_integration_step = int(T / delta_t)
    nb_simulation = 300
    seed = 42

    folder_path = f'./simulation/dof10_simulation/'
    create_folder_if_not_exists(folder_path)

    vy_noise = 0.1 # m/s
    psidt_noise = np.pi/180 # rad/s

    for n in range(nb_simulation):

        results = {'vy' : [], 'psidt':[], 'vy_' : [], 'psidt_':[], 'vx':[], 'alpha1':[]}

        X0 = [
                vehicle_parameters.x[n],
                vehicle_parameters.vx[n],
                vehicle_parameters.y[n],
                vehicle_parameters.vy[n],
                vehicle_parameters.z[n],
                vehicle_parameters.vz[n],
                vehicle_parameters.theta[n],
                vehicle_parameters.thetadt[n],
                vehicle_parameters.phi[n],
                vehicle_parameters.phidt[n],
                vehicle_parameters.psi[n],
                vehicle_parameters.psidt[n],
                vehicle_parameters.omega1[n],
                vehicle_parameters.omega2[n],
                vehicle_parameters.omega3[n],
                vehicle_parameters.omega4[n]
            ]
            
        U = np.asarray([
            vehicle_parameters.tf[n],
            vehicle_parameters.tf[n],
            vehicle_parameters.tr[n],
            vehicle_parameters.tr[n],
            vehicle_parameters.df[n],
            vehicle_parameters.dr[n]])

        model = DOF10SimplifiedPacejka(DOF10SimplifiedPacejka_parameters)

        results['vx'].append(X0[1])
        results['alpha1'].append(vehicle_parameters.df[n][0])

        results['psidt'].append(X0[11] + np.random.randn() * psidt_noise)
        results['vy'].append(X0[3]+ np.random.randn() * vy_noise)
        results['psidt_'].append(X0[11] )
        results['vy_'].append(X0[3] )

        Xdt = np.asarray(model.run(X0, U[:, 0])) 
        X = Xdt * delta_t + np.asarray(X0)

        for ii in tqdm(range(1, nb_integration_step)):
            
            results['vx'].append(X[1])
            results['alpha1'].append(vehicle_parameters.df[n][ii])
            
            results['psidt'].append(X[11] + np.random.randn() * psidt_noise)
            results['vy'].append(X[3] + np.random.randn() * vy_noise)

            results['psidt_'].append(X[11])
            results['vy_'].append(X[3])

            Xdt = np.asarray(model.run(X, U[:, ii]))
            X = Xdt * delta_t + np.asarray(X)

        data = pd.DataFrame(results)
        data.to_csv(folder_path+f'simulation_{n}.csv')
 

