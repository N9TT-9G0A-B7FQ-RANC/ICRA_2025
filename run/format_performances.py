import numpy as np
import pandas as pd

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

    mhe_performance = pd.read_csv(f'{training_folder}/mhe_performances/performances.csv')
    performance = pd.read_csv(f'{training_folder}/performances/performances.csv')

    future_temporal_horizons = [0.1, 0.2, 0.4, 0.8]
    
    versions = ['cnn', 'gru', 'transformer']
    past_temporal_horizons = [0.5, 1, 2, 4]
    datas = ['vanderpol_simulation', 'duffing_simulation', 'dof10_simulation']

    formated_results = {'version':[],
                        'data':[],
                        'past_temporal_horizon':[],
                        'future_temporal_horizon':[],
                        'computation_time':[],
                        'x0':[],
                        'x1':[]
    }

    for version in versions:
        for data in datas:
            for past_temporal_horizon in past_temporal_horizons:
                for future_temporal_horizon in future_temporal_horizons:
                    idx = filter_method(past_temporal_horizon_value=past_temporal_horizon ,future_temporal_horizon=future_temporal_horizon, data_value=data, version_value=version)
                    for state in ['x0', 'x1']:
                        res_mean = performance.iloc[idx][state].mean()
                        res_std = performance.iloc[idx][state].std()
                        formated_results[f'{state}'].append(f"{round(res_mean, 4)} $\pm$ {round(res_std, 4)}") # \pm {round(res_std, 4)}")

                    formated_results['version'].append(version)
                    formated_results['data'].append(data)
                    formated_results['past_temporal_horizon'].append(past_temporal_horizon)
                    formated_results['computation_time'].append(round(performance.iloc[idx]['computation_time'].mean(), 4))
                    formated_results['future_temporal_horizon'].append(future_temporal_horizon)

    for data in datas:
        for past_temporal_horizon in past_temporal_horizons:
            idx = filter_mhe(past_temporal_horizon, data)
            for state in ['x0', 'x1']:
                res_mean = mhe_performance.iloc[idx][state].mean()
                res_std = mhe_performance.iloc[idx][state].std()
                formated_results[f'{state}'].append(f"{round(res_mean, 4)}") # \pm {round(res_std, 4)}")

            formated_results['version'].append('mhe')
            formated_results['data'].append(data)
            formated_results['past_temporal_horizon'].append(past_temporal_horizon)
            formated_results['computation_time'].append(round(mhe_performance.iloc[idx]['computation_time'].mean()*10, 4))     
            formated_results['future_temporal_horizon'].append(np.nan)
    
    pd.DataFrame(formated_results).to_csv('./formated_results.csv')


    