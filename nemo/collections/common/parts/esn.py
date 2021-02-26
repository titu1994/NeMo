# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def esn(input_size: int, hidden_size: int, num_layers: int, sparsity: float, activation: str,) -> torch.nn.Module:
    """
    Utility function to provide unified interface to common LSTM RNN modules.

    Args:
        input_size: Input dimension.

        hidden_size: Hidden dimension of the RNN.

        num_layers: Number of RNN layers.
    Returns:
        A RNN module
    """
    return torch.jit.script(
        esn_rnn(  # torch.jit.script(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            sparsity=sparsity,
            activation=activation,
        )
    )


def esn_rnn(input_size: int, hidden_size: int, num_layers: int, sparsity: float, activation: str,) -> torch.nn.Module:
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    valid_activations = ['tanh', 'relu']
    if activation not in valid_activations:
        raise ValueError(f"Activation must be in {valid_activations}")

    return StackedRNN(
        num_layers,
        ESNLayer,
        first_layer_args=[ESNCell, input_size, hidden_size, sparsity, activation],
        other_layer_args=[ESNCell, hidden_size, hidden_size, sparsity, activation],
    )


class ESNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(ESNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: torch.Tensor, state: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ESNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, sparsity, activation):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity

        self.weight_ih = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.weight_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Make reservoir weights untrainable
        self.weight_ih.requires_grad_(False)
        self.weight_hh.requires_grad_(False)

        # Float scaling factors (initalized to 1)
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        self.rho = torch.nn.Parameter(torch.tensor(0.99))

        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        self.act_str = activation

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1.0  / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'readout' in name:
                # stdv = 1.0 / math.sqrt(self.hidden_size)
                # torch.nn.init.uniform_(weight, -stdv, stdv)
                torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5.0))
            elif 'gamma' in name:
                torch.nn.init.ones_(weight)
            elif 'rho' in name:
                torch.nn.init.constant_(weight, 0.99)
            else:
                stdv = 1.0  # / math.sqrt(self.hidden_size)
                torch.nn.init.uniform_(weight, -stdv, stdv)

        # compute hidden to hidden matrix spectral norm and scale weights
        eigen_values, _ = torch.eig(self.weight_hh.weight, eigenvectors=False)
        lambda_eig = eigen_values.abs().max()  # maximum real valued eigenvalue
        if lambda_eig > 0.0:
            self.weight_hh.weight.data = self.weight_hh.weight.data / lambda_eig

        # apply sparsity only to hidden_hh
        mask = torch.rand(
            *self.weight_hh.weight.shape, device=self.weight_hh.weight.device, dtype=self.weight_hh.weight.dtype, requires_grad=False
        )
        mask = mask <= self.sparsity
        self.weight_hh.weight.data = self.weight_hh.weight.data.masked_fill(mask, value=0.0)
        del mask

    def forward(self, input: torch.Tensor, state: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        hx = state[0]
        igates = self.gamma * self.weight_ih(input)
        hgates = torch.clamp_max(self.rho, 1.0) * self.weight_hh(hx)
        gates = igates + hgates

        hy = self.activation(gates)

        readout = self.readout(hy)
        # readout = self.activation(readout)

        return readout, (hy,)


def init_stacked_lstm(
    num_layers: int, layer: torch.nn.Module, first_layer_args: List, other_layer_args: List
) -> torch.nn.ModuleList:
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]
    return torch.nn.ModuleList(layers)


class StackedRNN(torch.nn.Module):
    def __init__(self, num_layers: int, layer: torch.nn.Module, first_layer_args: List, other_layer_args: List):
        super(StackedRNN, self).__init__()
        self.layers: torch.nn.ModuleList = init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args)

    def forward(
        self, input: torch.Tensor, states: Optional[Tuple[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:

        num_layers: int = len(self.layers)

        if states is None:
            batch = input.size(1)
            temp_states: Tuple[torch.Tensor] = (
                torch.zeros(
                    num_layers, batch, self.layers[0].cell.hidden_size, dtype=input.dtype, device=input.device
                ),
            )

            states = temp_states

        num_states: int = len(states)
        output_states: List[torch.Tensor] = []
        output = input
        for i, rnn_layer in enumerate(self.layers):
            # HARDCODE STATE TUPLE => 1 WRAP
            state: Tuple[torch.Tensor] = (states[0][i],)
            output, out_state = rnn_layer(output, state)
            # HARDCODE STATE TUPLE => 1 UNWRAP
            output_states.append(out_state[0])
            # i += 1
        # HARDCODE STATE TUPLE => 1 WRAP
        output_tensor: Tuple[torch.Tensor] = (torch.stack(output_states),)
        del output_states
        return output, output_tensor


def label_collate(labels, device=None):
    """Collates the label inputs for the rnn-t prediction network.
    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.
        device: Optional torch device to place the label on.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(f"`labels` should be a list or tensor not {type(labels)}")

    batch_size = len(labels)
    max_len = max(len(label) for label in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, : len(l)] = l
    labels = torch.tensor(cat_labels, dtype=torch.int64, device=device)

    return labels
