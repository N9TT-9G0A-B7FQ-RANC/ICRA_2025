import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import torch
from config import *
from tqdm import tqdm
import gc
from nn_architecture import observer_model
from utils import data_preprocessing, open_data, open_json, create_folder_if_not_exists
import argparse
import time
    
def get_observation(model, X, U, input_size, output_size, delay, sequence_duration, dt):
    """
    Generates observations using a state observer model.

    Args:
        model (torch.nn.Module): The state observer model.
        X (torch.Tensor): Input data representing the state of the system.
        U (torch.Tensor): Input data representing the control inputs.
        input_size (int): Size of the input data.
        output_size (int): Size of the output data (observation).
        delay (float): Time delay in the system.
        sequence_duration (float): Duration of each observation sequence.
        dt (float): Time step.

    Returns:
        torch.Tensor: Tensor containing the generated observations.

    This function generates observations by applying a state observer model to input sequences (X, U).
    It uses a sliding window approach with a specified time delay and sequence duration.
    The resulting observations are stored in a tensor and returned.
    """
    sequence_size = int(sequence_duration / dt) + 1
    space_between_element = int(delay / dt)
    in_idx = torch.arange(0, sequence_size, space_between_element).long().to(device)
    results = torch.zeros((X.shape[0], X.shape[1] - sequence_size, output_size))

    avg_computation_time = []
    # if device == 'cpu':
    for traj_idx in range(X.shape[0]):
        start_time = time.time()
        for i in range(0, nb_step - sequence_size):
            if version == 'gru':
                model.reset_hidden_state(X[0:1])
            indexed_X = torch.index_select(X[traj_idx:traj_idx+1], 1, i + in_idx).clone()
            indexed_U = torch.index_select(U[traj_idx:traj_idx+1], 1, i + in_idx).clone()
            pred = model.forward(indexed_X, indexed_U, None, batchsize = 1, mode = 'state_observer')
            results[traj_idx:traj_idx+1, i] = pred[0]
        end_time = time.time()
    avg_computation_time.append((end_time - start_time)/ (nb_step - sequence_size))

    # if device == 'cuda':
    #     start_time = time.time()
    #     for i in range(0, nb_step - sequence_size):
    #         if version == 'gru':
    #             model.reset_hidden_state(X[0:X.shape[0]])
    #         indexed_X = torch.index_select(X, 1, i + in_idx).clone()
    #         indexed_U = torch.index_select(U, 1, i + in_idx).clone()
    #         pred = model.forward(indexed_X, indexed_U, None, batchsize = X.shape[0], mode = 'state_observer')
    #         results[:, i] = pred
    #     end_time = time.time()
    # avg_computation_time.append((end_time - start_time)/ (nb_step - sequence_size))

    return results, np.mean(avg_computation_time)

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

    # training_name = "experiments"
    # training_parameters = "parameters"
    # start = 144
    # end = 145

    training_folder = f"./results/training_{training_name}/"
    training_parameters = pd.read_csv(f"{training_folder}/config.csv")
    plot = True

    device = 'cuda'
    # torch.set_num_threads(1)

    groundtruth = []
    noisy_groundtruth = []
    predictions = []
    unscented_predictions = []
    control = []

    nb_training = len(training_parameters)

    performances = {}
    for state in ['x0', 'x1']:
        performances[state] = []
    performances['computation_time'] = []
    
    for training_idx in tqdm(range(start, end, 1)):

        training_config = open_json(f"{training_folder}/training_config_{training_idx}.json")
            
        subsampling_dt = float(training_parameters['subsampling_dt'].values[training_idx])
        nb_hidden_layer = int(training_parameters['nb_layers'].values[training_idx])
        nb_neurones_per_layer = int(training_parameters['nb_neurones_per_layers'].values[training_idx])
        activation = str(training_parameters['activation'].values[training_idx])
        batchsize = int(training_parameters['batchsize'].values[training_idx])
        sequence_duration = float(training_parameters['past_temporal_horizon'].values[training_idx])
        delay = float(training_parameters['past_delay'].values[training_idx])
        futur_delay = float(training_parameters['future_temporal_horizon'].values[training_idx])
        state_configuration = str(training_parameters['state_configuration'].values[training_idx])
        training_data = str(training_parameters['data'].values[training_idx])
        state_configuration = str(training_parameters['state_configuration'].values[training_idx])
        max_sequence_duration = int(np.max(training_parameters['past_temporal_horizon']))
        data_dt = float(training_parameters['data_dt'].values[training_idx])
        version = str(training_parameters['version'].values[training_idx])
        use_input = bool(training_parameters['use_input'].values[training_idx])
        nb_trajectories = int(training_parameters['nb_data'].values[training_idx])
        data_path = f'./simulation/{training_data}'

        if training_idx == start:
            data_list = open_data(data_path, 'simulation', nb_trajectories)
        elif prev_training_data != training_data:
            data_list = open_data(data_path, 'simulation', nb_trajectories)
        prev_training_data = training_data


        sequence_size = int(sequence_duration/delay) + 1
        nb_evaluation_step = int(eval_trajectory_duration / futur_delay)
        nb_residual_blocks = int(futur_delay / subsampling_dt)

        train_trajectories_idx = training_config['training_idx']
        val_trajectories_idx = training_config['validation_idx']
        test_trajectories_idx = training_config['test_idx']
        state_variables = training_config['states_variables']
        in_variables = training_config['in_states_variables']
        out_variables = training_config['out_states_variables']
        control_variables = training_config['control_variables']

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

        nb_evaluation_step = int(eval_trajectory_duration / subsampling_dt)
        input_size = len(state_variables) + len(control_variables)
        output_size = len(state_variables)
        
        model = observer_model(
                nb_integration_steps = nb_residual_blocks,
                control_variables = control_variables,
                observed_states = in_variables,
                nb_hidden_layer = nb_hidden_layer,
                nb_neurones_per_hidden_layer = nb_neurones_per_layer,
                activation = activation,
                delay = delay,
                sequence_duration = sequence_duration,
                dt = subsampling_dt,
                prior_model = state_configuration,
                device = device,
                version = version,
                use_input = use_input
            )

        model.load_state_dict(torch.load(f'{training_folder}/best_model_{training_idx}.pt'))
        model = model.to(device)
        model.eval()

        nb_test_trajectories = len(X_noisy_in_list)
        max_lag = int(max_sequence_duration / subsampling_dt) + 1
        nb_lag = int(sequence_duration / subsampling_dt) + 1
        nb_step = int(eval_trajectory_duration / subsampling_dt) - max_lag + nb_lag - 1
        
        X_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        U_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_control)).to(device)
        X_out_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        
        X_noisy_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)
        U_noisy_in_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_control)).to(device)
        X_noisy_out_tensor = torch.zeros((nb_test_trajectories, nb_step, nb_state)).to(device)

        for idx, (X_noisy_in, U_noisy_in, X_in, U_in) in enumerate(zip(X_noisy_in_list, U_noisy_in_list, X_in_list, U_in_list)):
            X_in_tensor[idx] = torch.tensor(X_in[max_lag - nb_lag:])
            U_in_tensor[idx] = torch.tensor(U_in[max_lag - nb_lag:])
            X_noisy_in_tensor[idx] = torch.tensor(X_noisy_in[max_lag - nb_lag:])
            U_noisy_in_tensor[idx] = torch.tensor(U_noisy_in[max_lag - nb_lag:])
        
        X_pred, computation_time = get_observation(
            model,
            X_noisy_in_tensor[:, :, observed_variables_idx],
            U_noisy_in_tensor,
            input_size,
            output_size,
            delay, 
            sequence_duration, 
            subsampling_dt
        )

        groundtruth = X_in_tensor[:, nb_lag:].detach().cpu().numpy()
        noisy_groundtruth = X_noisy_in_tensor[:, nb_lag:].detach().cpu().numpy()
        predictions = X_pred.detach().cpu().numpy()

        RMSE = np.sqrt(np.mean(np.mean((groundtruth - predictions)**2, axis=1), axis=0))
        for idx, state in enumerate(['x0', 'x1']):
            performances[state].append(RMSE[idx])
        performances['computation_time'].append(computation_time)

        create_folder_if_not_exists(f'{training_folder}/performances/')
        np.save(f'{training_folder}/performances/prediction_{training_idx}', predictions)
        np.save(f'{training_folder}/performances/groundtruth_{training_idx}', groundtruth)
        np.save(f'{training_folder}/performances/noisy_groundtruth_{training_idx}', noisy_groundtruth)

        del X_in, U_in, model
        gc.collect()
        torch.cuda.empty_cache()

        results = pd.DataFrame(performances).to_csv(f'{training_folder}/performances/performances.csv')