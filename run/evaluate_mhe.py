import sys
sys.path.append('./')

from hilo_mpc import Model, MHE
import casadi as ca
from config import system_configuration
# from bokeh.io import output_notebook, show
from bokeh.plotting import figure
# from bokeh.layouts import gridplot
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time
# import matplotlib.pyplot as plt
from utils import data_preprocessing, open_data, open_json, save_to_json, create_folder_if_not_exists
from config import *
import pandas as pd
from tqdm import tqdm
import argparse

def create_hilo_model(model_choice, dt):

    if model_choice == 'vanderpol':

        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0', 'x1'], short_description=['x0', 'x1'])
        model.set_measurements(['x0'])

        # known parameters
        mu = 2
        
        # Unwrap states
        x0 = x[0]
        x1 = x[1]

        d_x0__dt = mu * (x0 - 1/3 * x0**3 - x1)
        d_x1__dt = 1/mu * x0

        model.set_measurement_equations(x0)
        model.set_dynamical_equations([d_x0__dt, d_x1__dt])
        model.setup(dt=dt)

    elif model_choice == 'duffing':
    
        model = Model(plot_backend='bokeh')
        x = model.set_dynamical_states(['x0', 'x1'], short_description=['x0', 'x1'])
        u = model.set_inputs('u')
        model.set_measurements(['x0'])

        # known parameters
        gamma = 0.002
        beta = 1
        alpha = 1

        # Unwrap states
        x0 = x[0]
        x1 = x[1]


        d_x0__dt = x1
        d_x1__dt = -gamma * x1 - alpha * x0 - beta * x0**3 + u

        model.set_measurement_equations(x0)
        model.set_dynamical_equations([d_x0__dt, d_x1__dt])
        model.setup(dt=dt)

    elif model_choice == 'dof2' or model_choice == 'dof10':
            
            parameter = {
                'g':9.81,
                'm':1500,
                'lf':2,
                'l':4.5,
                'iz':10000,
                'By1': 7.1,
                'Cy1': 1.3,
                'Dy1': 3678.75,
                'Ey1': -1.0,
                'By2': 7.1,
                'Cy2': 1.3,
                'Dy2': 3678.75,
                'Ey2': -1.0,
            }

            def get_slipping_angle(vx, vy, delta):
                return delta - ca.arctan(vy / vx)

            def get_lateral_force(parameters, x, index):
                By = parameters[f'By{index}']
                Cy = parameters[f'Cy{index}']
                Dy = parameters[f'Dy{index}']
                Ey = parameters[f'Ey{index}']
                return Dy * ca.sin(Cy * ca.arctan(By * x - Ey * (By * x - ca.arctan(By * x))))
        
            model = Model()
            x = model.set_dynamical_states(['vy', 'psidt'], short_description=['x0', 'x1'])
            u = model.set_inputs('vx', 'alpha1')
            model.set_measurements(['vy', 'psidt'])

            # Unwrap states
            vy = x[0]
            psidt = x[1]

            d1 = u[1] * 2
            vx = u[0]

            # known parameters
            m = parameter['m']
            lf = parameter['lf']
            lr = parameter['l'] - lf
            iz = parameter['iz']
            m_inv = 1 / m
            iz_inv = 1 / iz
    
            # Project on tire frame
            vxp1 = vx #* ca.cos(alpha1) + (vy + lf * psidt) * ca.sin(alpha1)
            vxp2 = vx #* ca.cos(d2) # + (vy - lr * psidt) * torch.sin(d2)
       
            vyp1 = (vy + lf * psidt) # * ca.cos(alpha1) - u[0] * ca.sin(alpha1) # + b
            vyp2 = (vy - lr * psidt) # - vx * torch.sin(d2) # + b

            # Compute lateral slip angles
            alpha1 = get_slipping_angle(vxp1, vyp1, d1)
            alpha2 = get_slipping_angle(vxp2, vyp2, 0)

            # Compute lateral tire forces
            fyp1 = get_lateral_force(parameter, alpha1, 1)
            fyp2 = get_lateral_force(parameter, alpha2, 2)

            # Project on carbody frame
            fy1 = fyp1 * ca.cos(d1) 
            fy2 = fyp2 # * torch.cos(d2)

            vydt = m_inv * (fy1 + fy2) - vx * ca.sin(d1) * psidt
            psidt2 = iz_inv * (lf * fy1 - lr * fy2)

            model.set_measurement_equations([vy, psidt])
            model.set_dynamical_equations([vydt, psidt2])
            model.setup(dt=dt)

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    # Add command-line options
    parser.add_argument("--training_name", type=str, help="Training name")
    parser.add_argument("--training_parameters", type=str, help="Training parameters")
    parser.add_argument("--start_index", type=int, help="Start training index")
    parser.add_argument("--end_index", type=int, help="End training idx")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the options
    training_name = args.training_name
    training_parameters = args.training_parameters
    start = args.start_index
    end = args.end_index
  
    training_folder = f"./results/training_{training_name}/"
    parameters_path = f"./training_parameters/{training_parameters}.csv"
    parameters = pd.read_csv(parameters_path)

    performances = {}
    for state in ['x0', 'x1']:
        performances[state] = []
    performances['computation_time'] = []

    for eval_idx in tqdm(range(start, end, 1)):

        subsampling_dt = float(parameters['subsampling_dt'].values[eval_idx])
        past_sequence_duration = float(parameters['past_temporal_horizon'].values[eval_idx])
        past_delay = float(parameters['past_delay'].values[eval_idx])
        state_configuration = str(parameters['state_configuration'].values[eval_idx])
        smoothing = bool(parameters['smoothing'].values[eval_idx])
        training_data = str(parameters['data'].values[eval_idx])
        data_dt = float(parameters['data_dt'].values[eval_idx])
        nb_trajectories = int(parameters['nb_data'].values[eval_idx])

        model_choice =  str(parameters['state_configuration'].values[eval_idx]) #training_data.split('_')[0]
        nb_evaluation_step = int(eval_trajectory_duration / subsampling_dt)

        data_path = f'./simulation/{training_data}'
        data_list = open_data(data_path, 'simulation', nb_trajectories)

        sequence_size = int(past_sequence_duration/past_delay) + 1

        if training_data == 'vanderpol_simulation':
            training_config = open_json(f"{training_folder}/training_config_0.json")
        elif training_data == 'duffing_simulation':
            training_config = open_json(f"{training_folder}/training_config_48.json")
        elif training_data == 'dof10_simulation':
            training_config = open_json(f"{training_folder}/training_config_96.json")

        train_trajectories_idx = training_config['training_idx']
        val_trajectories_idx = training_config['validation_idx']
        test_trajectories_idx = training_config['test_idx']

        in_variables = system_configuration[model_choice]['observed_state']
        out_variables = system_configuration[model_choice]['observed_state']
        control_variables = system_configuration[model_choice]['control']
        state_variables = system_configuration[model_choice]['state']

        observed_variables_idx = []
        non_observed_variables_idx = []
        for idx in range(len(state_variables)):
            for variable in in_variables:
                if variable == state_variables[idx] or f'{variable}_' == state_variables[idx]:
                    observed_variables_idx.append(idx)
                    break
        for idx in range(len(state_variables)):
                if idx not in observed_variables_idx:
                    non_observed_variables_idx.append(idx)

        train_data_list, val_data_list, test_data_list = [], [], []
        for idx in train_trajectories_idx:
            train_data_list.append(data_list[idx].copy())
        for idx in val_trajectories_idx:
            val_data_list.append(data_list[idx].copy())
        for idx in test_trajectories_idx:
            test_data_list.append(data_list[idx].copy())

        nb_in_state = len(in_variables)
        nb_out_state = len(out_variables)
        nb_state = len(state_variables)
        nb_control = len(control_variables)

        X_noisy_in_list, U_noisy_in_list, X_noisy_out_list = data_preprocessing(
                data_list = test_data_list.copy(),
                data_dt = data_dt,
                subsampling_dt = subsampling_dt,
                state_variables = state_variables,
                out_variables = state_variables,
                control_variables = control_variables,
                differentiate = False,
                smoothing = {},
            )

        X_in_list, U_in_list, X_out_list = data_preprocessing(
                data_list = test_data_list.copy(),
                data_dt = data_dt,
                subsampling_dt = subsampling_dt,
                state_variables = [f'{state}_' for state in state_variables],
                out_variables = [f'{state}_' for state in state_variables],
                control_variables = control_variables,
                differentiate = False,
                smoothing = {},
            )
        
        model = create_hilo_model(model_choice = model_choice, dt = subsampling_dt)

        estimations = []
        avg_times = []

        for idx, (X_noisy_in, U_noisy_in, X_in, U_in) in tqdm(enumerate(zip(X_noisy_in_list, U_noisy_in_list, X_in_list, U_in_list))):

            max_lag = int(max_sequence_duration / past_delay) + 1
            nb_lag = int(past_sequence_duration / past_delay) + 1
            nb_step = int(eval_trajectory_duration / subsampling_dt) - max_lag + nb_lag - 1

            X_in = X_in[max_lag - nb_lag:]
            U_in = U_in[max_lag - nb_lag:]
            X_noisy_in = X_noisy_in[max_lag - nb_lag:]
            U_noisy_in = U_noisy_in[max_lag - nb_lag:]

            if model_choice != 'dof2' or model_choice != 'dof10':
                unobserved_state_init = np.asarray(X_noisy_in_list)[ :, :, 1].min() + np.random.rand() * (np.asarray(X_noisy_in_list)[ :, :, 1].max() - np.asarray(X_noisy_in_list)[ :, :, 1].min())
                initial_cond = [X_noisy_in[0][0], unobserved_state_init]
                model.set_initial_conditions(x0 = initial_cond)
            else:
                initial_cond = X_noisy_in[0]
                model.set_initial_conditions(x0 = X_noisy_in[0])

            # Compute weights
            measurement_weights = [1/system_configuration[model_choice]['state_noise'][idx] for idx in observed_variables_idx]             
            process_weight =  [1/system_configuration[model_choice]['process_noise'][idx]**2 for idx in range(len(state_variables))]   
            

            # Setup the MHE
            mhe = MHE(model)
            mhe.quad_arrival_cost.add_states(weights=process_weight, guess=initial_cond)
            mhe.quad_stage_cost.add_measurements(weights=measurement_weights)
            mhe.quad_stage_cost.add_state_noise(weights=process_weight)
            mhe.horizon = nb_lag
            mhe.setup()

            estimation = []
                
            start_time = time.time()

            for i in range(0, nb_step):
                if model_choice == 'duffing':

                    model.simulate(u=U_noisy_in[i])
                    mhe.add_measurements(y_meas=X_noisy_in[i, observed_variables_idx][0], u_meas=U_noisy_in[i][0])

                elif model_choice == 'vanderpol':

                    model.simulate()
                    mhe.add_measurements(y_meas=X_noisy_in[i, observed_variables_idx][0])

                elif model_choice == 'dof2':
                    model.simulate(u=U_noisy_in[i])
                    mhe.add_measurements(y_meas=X_noisy_in[i, observed_variables_idx], u_meas=U_noisy_in[i])

                x_est, p0 = mhe.estimate()
                if x_est is not None:
                    estimation.append(np.asarray(x_est))

            end_time = time.time()

            avg_times.append((end_time - start_time)/(nb_step - sequence_size))
            estimations.append(estimation)

        predictions = np.asarray(estimations)[..., 0]
        groundtruth = np.asarray(X_in_list)[:, max_lag-1:]
        noisy_groundtruth = np.asarray(X_noisy_in_list)[:, max_lag-1:]

        RMSE = np.sqrt(np.mean(np.mean((groundtruth - predictions)**2, axis=1), axis=0))
        for idx, state in enumerate(['x0', 'x1']):
            performances[state].append(RMSE[idx])
        performances['computation_time'].append(np.mean(avg_times))
       
        create_folder_if_not_exists(f'{training_folder}/mhe_performances/')
        np.save(f'{training_folder}/mhe_performances/prediction_{eval_idx}', predictions)
        np.save(f'{training_folder}/mhe_performances/groundtruth_{eval_idx}', groundtruth)
        np.save(f'{training_folder}/mhe_performances/noisy_groundtruth_{eval_idx}', noisy_groundtruth)

        pd.DataFrame(performances).to_csv(f'{training_folder}/mhe_performances/performances.csv')