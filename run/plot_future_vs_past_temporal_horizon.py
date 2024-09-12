import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    mhe_performance = pd.read_csv(f'{training_folder}/mhe_performances/performances.csv')
    performance = pd.read_csv(f'{training_folder}/performances/performances.csv')

    
    versions = ['gru']
    past_temporal_horizons = [0.5, 1., 2., 4.]
    future_temporal_horizons = [0.1, 0.2, 0.4, 0.8]
    datas = ['vanderpol_simulation','duffing_simulation', 'dof10_simulation']

    formated_results = {'version':[],
                        'data':[],
                        'past_temporal_horizon':[],
                        'future_temporal_horizon':[],
                        'x0':[],
                        'x1':[],
                        'x0_std':[],
                        'x1_std':[]
    }

    for version in versions:
        for data in datas:
            for past_temporal_horizon in past_temporal_horizons:
                for future_temporal_horizon in future_temporal_horizons:
                    idx = filter_method(past_temporal_horizon_value=past_temporal_horizon ,future_temporal_horizon=future_temporal_horizon, data_value=data, version_value=version)
                    for state in ['x0', 'x1']:
                        res_mean = performance.iloc[idx][state].mean()
                        res_std = performance.iloc[idx][state].std()
                        formated_results[f'{state}'].append(res_mean)
                        formated_results[f'{state}_std'].append(res_std)

                    formated_results['version'].append(version)
                    formated_results['data'].append(data)
                    formated_results['past_temporal_horizon'].append(past_temporal_horizon)
                    formated_results['future_temporal_horizon'].append(future_temporal_horizon)

    # mhe_formated_results = {
    #                     'data':[],
    #                     'past_temporal_horizon':[],
    #                     'x0':[],
    #                     'x1':[]
    # }
    # for data in datas:
    #     for past_temporal_horizon in past_temporal_horizons:
    #         idx = filter_mhe(past_temporal_horizon, data)
    #         for state in ['x0', 'x1']:
    #             res_mean = mhe_performance.iloc[idx][state].mean()
    #             res_std = mhe_performance.iloc[idx][state].std()
    #             mhe_formated_results[f'{state}'].append(res_mean)

    #         mhe_formated_results['data'].append(data)
    #         mhe_formated_results['past_temporal_horizon'].append(past_temporal_horizon)
    
    # mhe_df = pd.DataFrame(mhe_formated_results)
    df = pd.DataFrame(formated_results)
    legend = True
    for data in datas:
        for version in versions:

            colors = ['royalblue', 'red', 'lime', 'magenta']

            plt.figure(figsize = (10, 10))
            for idx, past_temporal_horizon in enumerate(past_temporal_horizons):

                mask = (df['data'] == data) & (df['past_temporal_horizon'] == past_temporal_horizon) & (df['version'] == version)
                val = (df[mask]['x0'].values + df[mask]['x1'].values)/2
                std = (df[mask]['x0_std'].values + df[mask]['x1_std'].values)/2
                upper_bound = val + std * 0.5
                lower_bound = val - std * 0.5
                plt.plot(df[mask]['future_temporal_horizon'].values, val, color = colors[idx], label=f'$T_p$ = {past_temporal_horizon}')
                plt.fill_between(df[mask]['future_temporal_horizon'].values, lower_bound, upper_bound, alpha=0.1, color=colors[idx])

            if legend:   
                plt.legend(fontsize = 20)
                legend=False
            plt.tick_params(axis='both', labelsize=20)
            plt.xlabel('Future Temporal Horizon (seconds)', fontsize = 20)
            plt.ylabel('RMSE', fontsize = 20)
            plt.grid()
            plt.savefig(f'./figures/future_vs_past_temp/{version}_{data}.pdf')