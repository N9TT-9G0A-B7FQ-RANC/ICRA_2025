# Learning Direct Solutions in Moving Horizon Estimation with Deep Learning Methods

# General description of the method :

The proposed method is based on an end-to-end differentiable formulation of the Moving Horizon Estimation (MHE) optimization problem, allowing for the offline training of a Deep Learning model to provide state estimates that approximate the solution of the MHE optimization criterion.

The training process follows five steps:

1) The Deep Learning model, denoted as $\phi$, is fed with a past horizon of potentially partial and noisy measurements and outputs an initial state estimate $\mathbf{x}(t_0)$.

2) A state trajectory estimate is obtained by numerically integrating a known dynamical model $f$, which represents the underlying measured system, starting from the initial state estimate $\mathbf{x}(t_0)$.

3) The state trajectory estimate is applied through the observation function.

4) A loss is computed to measure the divergence between the estimated final trajectory $\hat{y}$ and the noisy system measurements $y$.

5) The gradient of the Deep Learning model parameters $\theta$ is computed from the loss function using an automatic differentiation algorithm. This allows the model to be optimized via gradient descent to predict state estimates that best align with the measured trajectories.

Once the training is complete, the Deep Learning model learns to estimate solutions that approximate the MHE criterion, without incurring the overhead costs related to numerical system integration and online solvers, as in the online formulation of the Moving Horizon Estimator.



![Method overview](./controllers_brief.svg)
<img src="./method_desc.svg">

# Installation 
Python 3.8.10 or higher

# Installation via Github
git clone git@github.com:N9TT-9G0A-B7FQ-RANC/ICRA_2025.git

Requirements
```bash
# Python 3.8.10
bokeh==3.1.1
casadi==3.6.6
HILO_MPC==1.0.3
matplotlib==3.6.2
numpy==1.24.4
pandas==1.5.2
torch==1.13.1
tqdm==4.64.1
```

After activating the virtual environment, you can install specific package requirements as follows
```bash
pip install -r requirements.txt
```

# Generate Dataset
To generate the required dataset for training the model, execute the following commands:

```bash
python ./system/simulate_vanderpol.py
python ./system/simulate_duffing_oscillator.py
python ./system/simulate_dof10.py
```

# Launch Training

Training configurations can be found in ./training_parameters/parameters.csv

Experiments are repeated 3 times for each parameter configurations and are ordered in parameters.csv file in the following order :

|      Method     |  Case study | Start index | End index |
|:---------------:|:-----------:|:-----------:|:---------:|
|       CNN       | Van der Pol |      0      |     47    |
|       CNN       |   Duffing   |      48     |     95    |
|       CNN       |    Dof10    |      96     |    144    |
|       GRU       | Van der Pol |     145     |    191    |
|       GRU       |   Duffing   |     192     |    239    |
|       GRU       |    Dof10    |     240     |    287    |
| CNN-Transformer | Van der Pol |     288     |    335    |
| CNN-Transformer |   Duffing   |     336     |    383    |
| CNN-Transformer |    Dof10    |     384     |    431    |


Experiments can be launched by running the following python scripts :

```bash
# CNN experiments 
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 0 --end_index 48
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 48 --end_index 96
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 96 --end_index 145

# GRU experiments
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 145 --end_index 192
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 192 --end_index 240
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 240 --end_index 288

# CNN-Transormer experiments
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 288 --end_index 336
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 336 --end_index 384
python ./run/train.py --training_name experiments --training_parameters parameters.csv --start_index 384 --end_index 432

# Or run all experiments in the same process
python ./run/train.py --training_name experiments --training_parameters parameters --start_index 0 --end_index 432
```

# Evaluate trained models
After training, performances of trained architectures can be evaluated in a similar way by running :

```bash
python ./run/evaluate.py --training_name experiments --training_parameters parameters --start_index 0 --end_index 432
```

# To evaluate the online Moving Horizon Estimator method :
```bash
python ./run/evaluate_mhe.py --training_name experiments --training_parameters parameters_mhe --start_index 0 --end_index 12
```

# To format all obtained results :
```bash
# Obtain all results under .csv format in current directory
python ./run/format_performances.py

# Plot state trajectories
python ./run/plot_state_estimations.py  

# Plot RMSE evolution for different combinations of Tf and Tp parameters
python ./run/plot_future_vs_past_temporal_horizon.py
```