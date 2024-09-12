import numpy as np

duffing_state_variables = ['x', 'y']
duffing_observed_state_variables = ['x']
duffing_control_variables = ['f']


vanderpol_state_variables = ['x', 'y']
vanderpol_observed_state_variables = ['x']
vanderpol_control_variables = []

system_configuration = {
    'vanderpol' : {
        'state' : vanderpol_state_variables,
        'observed_state' : vanderpol_observed_state_variables,
        'control' : vanderpol_control_variables,
        'state_noise':[1.5, 1.5],
        'process_noise':[0.01, 0.01]
    },
    'duffing': {
        'state' : duffing_state_variables,
        'observed_state' : duffing_observed_state_variables,
        'control' : duffing_control_variables,
        'state_noise':[0.45, 0.45],
        'process_noise':[0.01, 0.01]
    },
    'dof2':{
            'state':['vy', 'psidt'],
            'observed_state':['vy', 'psidt'],
            'control':['vx', 'alpha1'],
            'state_noise':[0.25, np.pi/180 * 1],
            'process_noise':[0.01, 0.01]
        },
    'dof10':{
            'state':['vy', 'psidt'],
            'observed_state':['vy', 'psidt'],
            'control':['vx', 'alpha1'],
            'state_noise':[0.25, np.pi/180 * 1],
            'process_noise':[0.01, 0.01]
        },
}

max_sequence_duration = 4.

nb_trajectories = 300

seed = 42
device = 'cuda:0'
train_set_pct = 0.7
val_set_pct = 0.2
test_set_pct = 0.1
eval_trajectory_duration = 20.
validation_frequency = 1
nb_epochs = 20

smoothing_parameters = {}