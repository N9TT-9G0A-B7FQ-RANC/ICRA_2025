import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import torch
import os
from utils import to_numpy, SimpleDataLoader, save_to_json, data_preprocessing, open_data, split_data
from config import *
from tqdm import tqdm
import gc
from nn_architecture import observer_model
import argparse
import time


def train_loop(
        model,
        optimizer, 
        dataLoader,
        standardize,
        std,
    ):

    mse_loss = []

    model.train()
   
    for X_in, U_in, X_out, U_out in tqdm(dataLoader):

        if version == 'gru':
            model.reset_hidden_state(X_in)

        X_out_pred, ic = model(
            X_in,
            U_in,
            U_out,
            batchsize = X_in.shape[0]
        )

        if standardize :
            X_out_std = X_out / std
            X_out_pred_std = X_out_pred / std
            ic_std = ic / std
            X_in_std = X_in / std
       
        se_std = (X_out_pred_std - X_out_std)**2
        se = torch.mean((X_out_pred - X_out)**2)
        loss = torch.mean(se_std)

        # Compute training loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        mse_loss.append(se.detach().cpu().item())

    mse_loss = np.asarray(mse_loss)
    rmse_loss = np.sqrt(np.mean(mse_loss, axis = 0))

    return rmse_loss

def validation_loop(
        model, 
        valDataLoaderOneStep,
    ):
    
    model.eval()
    mse_loss = []

    # One step evaluation
    for X_in, U_in, X_out, U_out in valDataLoaderOneStep:

        X_out_pred, ic = model(
            X_in,
            U_in,
            U_out,
            batchsize = X_in.shape[0]
        )
        se = torch.mean((X_out_pred - X_out)**2)
        mse_loss.append(se.detach().cpu().item())

    mse_loss = np.asarray(mse_loss)
    rmse_loss = np.sqrt(np.mean(mse_loss, axis = 0))

    return rmse_loss

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
    start_idx = args.start_index
    end_idx = args.end_index

    # training_name = 'experiments'
    # training_parameters = 'parameters'
    # start_idx = 1519
    # end_idx = 1520

    # training_name = 'test_gru'
    # training_parameters = 'parameters_test_annex'
    # start_idx = 256
    # end_idx = 320

    training_folder = f"./results/training_{training_name}/"
    parameters_path = f"./training_parameters/{training_parameters}.csv"

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    parameters = pd.read_csv(parameters_path)
    parameters.to_csv(f'{training_folder}/config.csv')
    
    assert end_idx <= len(parameters)

    data_open = True

    for training_idx in range(start_idx, end_idx):
        
        # Open training parameters
        subsampling_dt = float(parameters['subsampling_dt'].values[training_idx])
        nb_hidden_layer = int(parameters['nb_layers'].values[training_idx])
        nb_neurones_per_layer = int(parameters['nb_neurones_per_layers'].values[training_idx])
        activation = str(parameters['activation'].values[training_idx])
        batchsize = int(parameters['batchsize'].values[training_idx])
        past_sequence_duration = float(parameters['past_temporal_horizon'].values[training_idx])
        future_sequence_duration = float(parameters['future_temporal_horizon'].values[training_idx])
        past_delay = float(parameters['past_delay'].values[training_idx])
        future_delay = float(parameters['future_delay'].values[training_idx])
        state_configuration = str(parameters['state_configuration'].values[training_idx])
        # smoothing = bool(parameters['smoothing'].values[training_idx])
        version = str(parameters['version'].values[training_idx])
        use_input = bool(parameters['use_input'].values[training_idx])
        nb_epochs = int(parameters['nb_epochs'].values[training_idx])
        nb_trajectories = int(parameters['nb_data'].values[training_idx])

        training_data = str(parameters['data'].values[training_idx])

        shuffle = True
        standardize = True
        learning_rate = 1e-3
  
        data_dt = float(parameters['data_dt'].values[training_idx])

        np.random.seed(seed)
        data_path = f'./simulation/{training_data}'

        # if data_open:
        data_list = open_data(data_path, 'simulation', nb_trajectories)
        data_open = False

        (train_data_list, train_trajectories_idx, 
         val_data_list, val_trajectories_idx, 
         test_data_list, test_trajectories_idx) = split_data(data_list, nb_trajectories, shuffle, train_set_pct, val_set_pct)

        sequence_size = int(past_sequence_duration/past_delay) + 1
        nb_evaluation_step = int(eval_trajectory_duration / subsampling_dt)
        nb_residual_blocks = int(future_sequence_duration / future_delay)

        in_variables = system_configuration[state_configuration]['observed_state']
        out_variables = system_configuration[state_configuration]['observed_state']
        control_variables = system_configuration[state_configuration]['control']
        state_variables = system_configuration[state_configuration]['state']
 
        model = observer_model(
                nb_integration_steps = nb_residual_blocks,
                control_variables = control_variables,
                observed_states = in_variables,
                nb_hidden_layer = nb_hidden_layer,
                nb_neurones_per_hidden_layer = nb_neurones_per_layer,
                activation = activation,
                delay = past_delay,
                sequence_duration = past_sequence_duration,
                dt = subsampling_dt,
                prior_model = state_configuration,
                device = device,
                version = version,
                use_input = use_input
            )
             
        train_in_state, train_in_control, train_out_state = data_preprocessing(
            data_list = train_data_list.copy(),
            data_dt = data_dt,
            subsampling_dt = subsampling_dt,
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = {},
            smoothing_parameters = smoothing_parameters
        )

        val_in_state, val_in_control, val_out_state = data_preprocessing(
            data_list = val_data_list.copy(),
            data_dt = data_dt,
            subsampling_dt = subsampling_dt,
            state_variables = in_variables,
            out_variables = out_variables,
            control_variables = control_variables,
            differentiate = False,
            smoothing = {},
            smoothing_parameters = smoothing_parameters
        )
        
        # Instantiate dataloader
        trainDataLoader = SimpleDataLoader(
            train_in_states = train_in_state.copy(),
            train_in_controls = train_in_control.copy(),
            train_out_states = train_out_state.copy(),
            batchsize = batchsize,
            past_sequence_duration = past_sequence_duration,
            future_sequence_duration = future_sequence_duration,
            past_delay = past_delay,
            future_delay = future_delay,
            dt = subsampling_dt,
            shuffle = shuffle,
            device = device
        )

        valDataLoader = SimpleDataLoader(
            train_in_states = val_in_state.copy(),
            train_in_controls = val_in_control.copy(),
            train_out_states = val_out_state.copy(),  
            batchsize = batchsize,
            past_sequence_duration = past_sequence_duration,
            future_sequence_duration = future_sequence_duration,
            past_delay = past_delay,
            future_delay = future_delay,
            dt = subsampling_dt,
            shuffle = shuffle,
            device = device
        )

        std = np.asarray(train_out_state).reshape(np.asarray(train_out_state).shape[0] * np.asarray(train_out_state).shape[1], len(out_variables)).std(axis=0)
        std = torch.tensor([std]).float().requires_grad_(False).to(device)

        training_config = {
            'in_states_variables' : list(in_variables),
            'out_states_variables': list(out_variables),
            'states_variables': list(state_variables),
            'control_variables':list(control_variables),
            'training_idx' : [int(idx) for idx in train_trajectories_idx],
            'validation_idx': [int(idx) for idx in val_trajectories_idx],
            'test_idx' : [int(idx) for idx in test_trajectories_idx],
        }

        save_to_json(
            training_config,
            f'{training_folder}/training_config_{training_idx}.json'
        )

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        best_loss = np.inf

        training_results = {}
        for state in out_variables:
            training_results[f"training_rmse_{state}"] = []
            training_results[f"one_step_validation_rmse_{state}"] = []
            training_results[f"multi_step_validation_rmse_{state}"] = []
        training_results[f"mean_position_error"] = []

        save_to_json(
            training_results,
            f'{training_folder}/training_results_{training_idx}.json'
            )
        estimated_time = []
        loss_val = [np.inf, np.inf, np.inf]
        for epochs in range(nb_epochs):
            start_time = time.time()
            train_loss = train_loop(
                model,
                optimizer,
                trainDataLoader,
                standardize,
                std,
            )
            end_time = time.time()
            estimated_time.append(end_time - start_time)
            print(f'Epoch : {epochs}, train loss : {np.mean(train_loss)}, time : {(np.mean(estimated_time) * (nb_epochs - epochs))/60}')
            print()

            if epochs % validation_frequency == 0:

                val_loss = validation_loop(
                    model,
                    valDataLoader,
                )

                print(f'Epoch : {epochs} | train loss : {np.mean(train_loss):.4f} | val loss : {np.mean(val_loss):.4f}')
                print()

                # for idx, state in enumerate(out_variables):
                #     training_results[f"training_rmse_{state}"].append(float(train_loss[idx]))
                #     training_results[f"one_step_validation_rmse_{state}"].append(float(val_loss[idx]))
                loss_val.append(best_loss)
                # if np.mean(val_loss) < best_loss:
                #     best_loss = np.mean(val_loss)
                torch.save(model.state_dict(), f'{training_folder}/best_model_{training_idx}.pt')
                # if best_loss >= loss_val[-2]:
                #     break
                # scheduler.step(val_loss)

        save_to_json(
            training_results,
            f'{training_folder}/training_results_{training_idx}.json'
            )
        
        del trainDataLoader, valDataLoader
        del model
        del train_in_state, train_in_control, train_out_state,
        del val_in_state, val_in_control, val_out_state
        del train_data_list, val_data_list, test_data_list

        gc.collect()
        torch.cuda.empty_cache()
        del data_list
    