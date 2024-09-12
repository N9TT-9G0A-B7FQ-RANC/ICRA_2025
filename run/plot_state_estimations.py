import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def filter_mhe(past_temporal_horizon_value, data_value):
    past_temporal_horizon_filter = mhe_parameters['past_temporal_horizon'] == past_temporal_horizon_value
    data_filter = mhe_parameters['data'] == data_value
    mask = past_temporal_horizon_filter & data_filter
    return mhe_parameters[mask].index.values

def filter_method(past_temporal_horizon_value, future_temporal_horizon, data_value, version_value):
    past_temporal_horizon_filter = parameters['past_temporal_horizon'] == past_temporal_horizon_value
    data_filter = parameters['data'] == data_value
    future_temporal_horizon_filter = parameters['future_temporal_horizon'] == future_temporal_horizon
    version = parameters['version'] == version_value
    mask = past_temporal_horizon_filter & data_filter & future_temporal_horizon_filter & version
    return parameters[mask].index.values

if __name__ == '__main__':
    
    training_name = 'experiments'

    training_folder = f"./results/training_{training_name}/"

    parameters_path = f"./training_parameters/parameters.csv"
    mhe_parameters_path = f"./training_parameters/parameters_mhe.csv"
    
    parameters = pd.read_csv(parameters_path)
    mhe_parameters = pd.read_csv(mhe_parameters_path)

    data_values = ['vanderpol_simulation', 'duffing_simulation', 'dof10_simulation']

    for data_value in data_values:
        
        plot_legend=True


        if data_value in ['vanderpol_simulation', 'duffing_simulation']:
            past_temporal_horizon = 4
            mhe_past_temporal_horizon = 4.
            future_temporal_horizon = 0.8
        else:
            past_temporal_horizon = 4
            future_temporal_horizon = 0.1
            mhe_past_temporal_horizon = 0.5

        print(mhe_past_temporal_horizon)
        mhe_idx = filter_mhe(past_temporal_horizon_value = mhe_past_temporal_horizon, data_value = data_value)[0]
        method_idx = filter_method(past_temporal_horizon_value = past_temporal_horizon, future_temporal_horizon = future_temporal_horizon, data_value=data_value, version_value='gru')

        mhe_prediction = np.load(f"{training_folder}/mhe_performances/prediction_{mhe_idx}.npy")
        
        traj_idx = 0

        # Specify the width and height ratio
        width = 10  # Width in inches
        ratio = 0.75  # Desired ratio (height/width)

        # Calculate the height based on the ratio
        height = width * ratio

        plt.figure(figsize = (10, 10))
        
        for _, idx in enumerate(method_idx): # 
            prediction = np.load(f"{training_folder}/performances/prediction_{idx}.npy")
            noisy_groundtruth = np.load(f"{training_folder}/performances/noisy_groundtruth_{idx}.npy")
            groundtruth = np.load(f"{training_folder}/performances/groundtruth_{idx}.npy")
            if plot_legend:
                plt.plot(np.arange(0, len(prediction[traj_idx, :, 0])) * 0.02, prediction[traj_idx, :, 0], c='royalblue', alpha = 1, label = 'Prediction')
            else:
                plt.plot(np.arange(0, len(prediction[traj_idx, :, 0])) * 0.02, prediction[traj_idx, :, 0], c='royalblue', alpha = 1)
            break
        plt.plot(np.arange(0, len(noisy_groundtruth[traj_idx, 1:, 0])) * 0.02, noisy_groundtruth[traj_idx, 1:, 0], c='grey', alpha=0.2, label = 'Groundtruth with noise')
        plt.plot(np.arange(0, len(groundtruth[traj_idx, 1:, 0])) * 0.02, groundtruth[traj_idx, 1:, 0], c='k', label = 'Groundtruth')
        plt.plot(np.arange(0, len(mhe_prediction[traj_idx, 1:, 0])) * 0.02, mhe_prediction[traj_idx, 1:, 0], c='r', alpha = 1, label = 'MHE prediction')

        plt.tick_params(axis='both', labelsize=20)
        plt.ylabel('$x_0$', fontsize = 20)
        plt.xlabel('Time (seconds)', fontsize = 20)
        if plot_legend:
            plt.legend(fontsize = 20, loc='lower left')
            plot_legend = False
        plt.gcf().set_size_inches(width, height)
        plt.grid()
        plt.savefig(f'./figures/state_estimation/{data_value}_x0_{past_temporal_horizon}.pdf')
        plt.close()
        
        plt.figure()
        
        for _, idx in enumerate(method_idx):
            prediction = np.load(f"{training_folder}/performances/prediction_{idx}.npy")
            noisy_groundtruth = np.load(f"{training_folder}/performances/noisy_groundtruth_{idx}.npy")
            groundtruth = np.load(f"{training_folder}/performances/groundtruth_{idx}.npy")
            if _ == 0:
                plt.plot(np.arange(0, len(prediction[traj_idx, :, 1])) * 0.02, prediction[traj_idx, :, 1], c='royalblue', alpha = 1, label = 'Prediction')
            else:
                plt.plot(np.arange(0, len(prediction[traj_idx, :, 1])) * 0.02, prediction[traj_idx, :, 1], c='royalblue', alpha = 1)
            break
        if data_value != 'duffing_simulation_c_bis':
            plt.plot(np.arange(0, len(noisy_groundtruth[traj_idx, 1:, 1])) * 0.02, noisy_groundtruth[traj_idx, 1:, 1], c='grey', alpha=0.2, label = 'Groundtruth with noise')
        plt.plot(np.arange(0, len(groundtruth[traj_idx, 1:, 1])) * 0.02, groundtruth[traj_idx, 1:, 1], c='k', label = 'Groundtruth')
        plt.plot(np.arange(0, len(mhe_prediction[traj_idx, 1:, 1])) * 0.02, mhe_prediction[traj_idx, 1:, 1], c='r', alpha = 1, label = 'MHE Prediction')
        plt.tick_params(axis='both', labelsize=20)
        plt.ylabel('$x_1$', fontsize = 20)
        plt.xlabel('Time (seconds)', fontsize = 20)
        if plot_legend:
            plt.legend(fontsize = 20, loc='lower left')
            plot_legend = False
        plt.gcf().set_size_inches(width, height)
        plt.grid()
        plt.savefig(f'./figures/state_estimation/{data_value}_x1_{past_temporal_horizon}.pdf')
        plt.close()