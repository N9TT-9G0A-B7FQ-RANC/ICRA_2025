import torch
from torch import Tensor
from collections import OrderedDict
from .pytorch_system import vanderpol, duffing_oscillator, dof2
    
class Mlp(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.

    Args:
        input_size (int): The size of the input feature.
        nb_hidden_layer (int): The number of hidden layers.
        nb_neurons_per_hidden_layer (int): The number of neurons in each hidden layer.
        output_size (int): The size of the output layer.
        activation (str): The activation function to be used ('relu', 'elu', 'sigmoid', 'tanh', 'softplus').

    Attributes:
        activation (torch.nn.Module): The activation function used in the hidden layers.
        layers (torch.nn.Sequential): The sequence of layers in the MLP.

    Example:
    ```
    mlp = MLP(input_size=64, nb_hidden_layer=2, nb_neurons_per_hidden_layer=128, output_size=10, activation='relu')
    output = mlp(input_data)
    ```

    """

    def __init__(
            self,
            input_size: int,
            nb_hidden_layer: int,
            nb_neurons_per_hidden_layer: int,
            output_size: int,
            activation: str,
        ):

        super(Mlp, self).__init__()

        layers = [input_size] + [nb_neurons_per_hidden_layer] * nb_hidden_layer + [output_size]
    
        # set up layer order dict
        if activation == 'none':
            self.activation = torch.nn.Identity
        if activation == 'relu':
            self.activation = torch.nn.ReLU
        if activation == 'elu':
            self.activation = torch.nn.ELU
        if activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'softplus':
            self.activation = torch.nn.Softplus

        depth = len(layers) - 1
        layer_list = list()
        for i in range(depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
            # layer_list.append(('dropout_%d' % i, torch.nn.Dropout(p = 0.1)))
            
        layer_list.append(
            ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, X):
        return self.layers(X)
    
class Nn_base(torch.nn.Module):

    def __init__(
            self,
        ):

        super(Nn_base, self).__init__()

    def set_z_scale(self, z_scale):
        self.register_buffer('z_scaling', torch.tensor(z_scale))
    
    def set_std(self, std):
        self.register_buffer('std', torch.tensor([std]).float())

    def set_mean(self, mean):
        self.register_buffer('mean', torch.tensor([mean]).float())

    def set_delay(self, delay):
        self.register_buffer('delay', torch.tensor(delay).float())

    def set_sequence_duration(self, sequence_duration):
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())

    def set_dt(self, dt):
        self.register_buffer('dt', torch.tensor([dt]).float())

        
# class Mlp_narx(Nn_base):

#     def __init__(
#             self, 
#             input_size,
#             nb_hidden_layer,
#             nb_neurones_per_hidden_layer,
#             output_size,
#             activation,
#         ):

#         super(Mlp_narx, self).__init__()

#         self.fc = Mlp(
#             input_size,
#             nb_hidden_layer,
#             nb_neurones_per_hidden_layer,
#             output_size,
#             activation,
#         )
        
#     def forward(self, X, U):
#         X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
#         U = U.reshape(U.shape[0], U.shape[1] * U.shape[2])
#         out = self.fc(torch.concat((X, U), dim=1))
#         return out 

class Mlp_rnn(Nn_base):
    """
    Multi-Layer Perceptron with Recurrent Neural Network (RNN) layers.

    Args:
        input_size (int): The size of the input feature.
        nb_hidden_layer (int): The number of hidden layers in the MLP.
        nb_neurones_per_hidden_layer (int): The number of neurons in each hidden layer of the MLP.
        output_size (int): The size of the output layer.
        activation (str): The activation function to be used ('relu', 'elu', 'sigmoid', 'tanh', 'softplus').
        recurrent_cell_type (str): The type of recurrent cell to use ('RNN', 'GRU', or 'LSTM').
        reccurrent_hidden_dim (int): The number of hidden units in the recurrent layer.
        nb_recurrent_layer (int): The number of recurrent layers.
        dt (float): The time step.
        sequence_duration (float): The duration of a sequence.
        delay (float): The delay parameter.
        std (float): Standard deviation.
        mean (float): Mean value.

    Attributes:
        recurrent_cell_type (str): The type of recurrent cell used.
        rnn (Union[torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM]): The recurrent cell layer.
        fc (MLP): The multi-layer perceptron.

    Example:
    ```
    model = MLP_RNN(input_size=64, nb_hidden_layer=2, nb_neurones_per_hidden_layer=128,
                    output_size=10, activation='relu', recurrent_cell_type='LSTM',
                    reccurrent_hidden_dim=64, nb_recurrent_layer=1, dt=0.01,
                    sequence_duration=1.0, delay=0.1, std=1.0, mean=0.0)
    prediction = model.forward(X, U)
    ```

    """

    # def __init__(self,
    #         input_size: int,
    #         nb_hidden_layer: int,
    #         nb_neurones_per_hidden_layer: int,
    #         output_size: int,
    #         activation: str,
    #         recurrent_cell_type: str,
    #         recurrent_hidden_dim: int,
    #         nb_recurrent_layer: int,
    #         dt: float,
    #         sequence_duration: float,
    #         delay: float,
    #     ):

    #     self.recurrent_cell_type = recurrent_cell_type
    #     self.recurrent_hidden_dim = recurrent_hidden_dim
    #     self.nb_recurrent_layer = nb_recurrent_layer

    #     super(Mlp_rnn, self).__init__()

    #     if recurrent_cell_type == 'rnn':
    #         self.rnn = torch.nn.RNN(
    #                 input_size, 
    #                 recurrent_hidden_dim, 
    #                 nb_recurrent_layer, 
    #                 batch_first=True
    #             )
    #     if recurrent_cell_type == 'gru':
    #         self.rnn = torch.nn.GRU(
    #                 input_size, 
    #                 recurrent_hidden_dim, 
    #                 nb_recurrent_layer, 
    #                 batch_first=True
    #             )
    #     if recurrent_cell_type == 'lstm':
    #         self.rnn = torch.nn.LSTM(
    #                 input_size, 
    #                 recurrent_hidden_dim, 
    #                 nb_recurrent_layer, 
    #                 batch_first=True
    #             )
        
    #     self.fc = Mlp(
    #         recurrent_hidden_dim,
    #         nb_hidden_layer,
    #         nb_neurones_per_hidden_layer,
    #         output_size,
    #         activation,
    #     )

    # def reset_state(
    #         self, 
    #         batchsize: int,
    #         device
    #     ):
    #     if self.recurrent_cell_type == 'lstm':
    #         self.c = torch.zeros(self.nb_recurrent_layer, batchsize, self.recurrent_hidden_dim).to(device).float()
    #     self.h = torch.zeros(self.nb_recurrent_layer, batchsize, self.recurrent_hidden_dim).to(device).float()
        
    # def forward(
    #         self,
    #         X: Tensor,
    #         U: Tensor
    #     ) -> Tensor:

    #     self.reset_state(X.shape[0], X.device)
        
    #     if self.recurrent_cell_type == 'lstm':
    #         out, (self.h, self.c) = self.rnn(torch.concat((X, U), dim=2), (self.h, self.c))
    #         self.h = self.h.detach()
    #         self.c = self.c.detach()
    #     else:
    #         out, self.h = self.rnn(torch.concat((X, U), dim=2), self.h)
    #         self.h = self.h.detach()

    #     out = out[:, -1, :]
    #     out = self.fc(out)
    
    #     return out
    
class observer_model(Nn_base):
    
    def __init__(
            self,
            nb_integration_steps,
            control_variables,
            observed_states,
            nb_hidden_layer,
            nb_neurones_per_hidden_layer,
            activation,
            delay,
            sequence_duration,
            dt,
            prior_model,
            version,
            device,
            use_input,
        ):
        
        super(observer_model, self).__init__()
        self.device = device
        self.nb_integration_steps = nb_integration_steps

        self.register_buffer('z_scaling', torch.tensor(False))
        self.register_buffer('std', torch.tensor([[1] * len(observed_states)]).float())
        self.register_buffer('mean', torch.tensor([[0] * len(observed_states)]).float())

       
        self.version = version
        if prior_model == 'vanderpol':
            self.prior_model = vanderpol()
         
        if prior_model == 'duffing':
            self.prior_model = duffing_oscillator()

        if prior_model == 'dof2':
            self.prior_model = dof2()
           
        observed_states_idx = []
        unobserved_states_idx = []
        for state in observed_states:
            for idx in range(len(self.prior_model.state_variables)):
                if self.prior_model.state_variables[idx] == state:
                    observed_states_idx.append(idx)
                    break
        for idx in range(len(self.prior_model.state_variables)):
            if idx not in observed_states_idx:
                unobserved_states_idx.append(idx)
        self.observed_states_idx = torch.tensor(observed_states_idx).int()
        self.unobserved_states_idx = torch.tensor(unobserved_states_idx).int()

        sequence_size = int(sequence_duration/delay) + 1
        input_size = len(observed_states) * sequence_size + len(control_variables) * sequence_size
        output_size = len(self.prior_model.state_variables)

        self.register_buffer('input_size', torch.tensor(input_size))
        self.register_buffer('output_size', torch.tensor(output_size))
        self.register_buffer('delay', torch.tensor(delay).float())
        self.register_buffer('sequence_duration', torch.tensor(sequence_duration).float())
        self.register_buffer('dt', torch.tensor([dt]).float())

        self.include_control = use_input
        if self.include_control:
            input_size = len(observed_states) + len(control_variables)
        else:
            input_size = len(observed_states)

        if version == 'mlp':
            self.model = Mlp(
                input_size = sequence_size * (len(observed_states) + len(control_variables)),
                nb_hidden_layer = int(nb_hidden_layer),
                nb_neurons_per_hidden_layer = int(nb_neurones_per_hidden_layer),
                output_size = len(self.prior_model.state_variables),
                activation = activation,
            )

        if version == 'cnn':

            kernel_size = 7
            stride = 2
            padding = 3
            output_size = 2

            self.conv1 = torch.nn.Conv1d(
                input_size, 
                output_size, 
                kernel_size, 
                stride, 
                padding)
            
            self.bn1 = torch.nn.BatchNorm1d(2)

            if sequence_size/stride - sequence_size//stride == 0:
                input_size = sequence_size // stride
            else:
                input_size = sequence_size // stride + 1

            self.model = Mlp(
                input_size = input_size*2,
                nb_hidden_layer = int(nb_hidden_layer),
                nb_neurons_per_hidden_layer = int(nb_neurones_per_hidden_layer),
                output_size = len(self.prior_model.state_variables),
                activation = activation,
            )

        if version == 'gru':
            
            self.fc = Mlp(
                input_size = sequence_size * input_size,
                nb_hidden_layer = 1,
                nb_neurons_per_hidden_layer = 8,
                output_size = sequence_size * input_size,
                activation = 'relu',
            )

            self.bn1 = torch.nn.BatchNorm1d(sequence_size)
            self.bn2 = torch.nn.BatchNorm1d(sequence_size)
            self.hidden_size = 8
            self.num_layers = 2
            self.gru = torch.nn.GRU(
                input_size = input_size, 
                hidden_size = self.hidden_size, 
                num_layers = self.num_layers, 
                batch_first = True,
                dropout = 0.1)
            
            self.hidden = None  # Initialize the hidden state variable

            self.model = torch.nn.Linear(self.hidden_size, 2)

        self.dropout = torch.nn.Dropout(p = 0.2)
        self.activation = torch.nn.Tanh()

        if version == 'transformer':

            kernel_size = 7
            stride = 1
            padding = 3
            output_size = 8

            self.conv1 = torch.nn.Conv1d(
                input_size, 
                output_size, 
                kernel_size, 
                stride, 
                padding)
            
            self.bn1 = torch.nn.BatchNorm1d(output_size)
            self.bn2 = torch.nn.BatchNorm1d(8)

            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model = output_size, 
                nhead = 1, 
                dim_feedforward = 8
                )
            
            self.transformer = torch.nn.TransformerEncoder(
                encoder_layer, 
                num_layers = 2
                )
            
            self.model  = torch.nn.Linear(sequence_size * 8, len(self.prior_model.state_variables))

    def reset_hidden_state(self, x):
        self.hidden = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    def interpolate_control(self, nb_step, current_input, next_input, subintegration_steps):
        return current_input + ((next_input - current_input) / subintegration_steps) * nb_step

    def forward(
            self,
            X_observed, 
            U,
            U_out,
            batchsize,
            mode = 'forward_prediction'):
        
        if self.version == 'cnn':
            if self.include_control:
                X = self.conv1(torch.cat((X_observed, U), dim=-1).permute(0, 2, 1))
            else:
                X = self.conv1(X_observed.permute(0, 2, 1))

            X = self.bn1(X)
            X_estimated = self.activation(X)
            X = self.dropout(X)
            X_estimated = self.model(X.flatten(1))

        if self.version == 'gru':
            if self.include_control:
                X = torch.cat((X_observed, U), dim=-1)
            else:
                X = X_observed
            X = self.fc(X.flatten(1)).reshape(X.shape)
            X = self.bn1(X)
            X, self.hidden = self.gru(X, self.hidden)
            X = self.bn2(X)
            self.hidden = self.hidden.detach()
            X_estimated = self.model(X[:, -1])

        if self.version == 'transformer':
            if self.include_control:
                X = self.conv1(torch.cat((X_observed, U), dim=-1).permute(0, 2, 1))
            else:
                X = self.conv1(X_observed.permute(0, 2, 1))

            X = self.activation(X)
            X = self.bn1(X)
            
            X = self.transformer(X.permute(0, 2, 1))
            X = self.bn2(X.permute(0, 2, 1))

            X_estimated = self.model(X.flatten(1))

        ic = X_estimated

        if mode == 'forward_prediction':
            outs = torch.zeros(batchsize, self.nb_integration_steps, len(self.prior_model.state_variables)).to(self.device)
            for i in range(self.nb_integration_steps):
                dt = self.dt / 5
                for _ in range(5):
                    if i == 0:
                        U_current = U[:, -1].unsqueeze(1)
                        U_next = U_out[:, 0].unsqueeze(1)
                    else:
                        U_current = U_out[:, i-1].unsqueeze(1)
                        U_next = U_out[:, i].unsqueeze(1)

                    U_in = self.interpolate_control(nb_step=_, 
                                                current_input=U_current,
                                                next_input=U_next,
                                                subintegration_steps=5)
                    
                    X_estimated = self.prior_model(X_estimated, U_in) * dt + X_estimated
                outs[:, i] = X_estimated.clone()
            return outs[:, :, self.observed_states_idx.long()], ic
            
        elif mode == 'state_observer':
            return X_estimated