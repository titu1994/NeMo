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


def esn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
) -> torch.nn.Module:
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
            activation=activation,
        )
    )


def esn_rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    activation: str,
) -> torch.nn.Module:
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""
    # The following are not implemented.
    valid_activations = ['tanh', 'relu']
    if activation not in valid_activations:
        raise ValueError(f"Activation must be in {valid_activations}")

    return StackedRNN(
        num_layers,
        ESNLayer,
        first_layer_args=[ESNCell, input_size, hidden_size, activation],
        other_layer_args=[ESNCell, hidden_size, hidden_size, activation],
    )


class ESNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args):
        super(ESNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ESNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, activation):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))

        # Float scaling factors (initalized to 1)
        self.rho = torch.nn.Parameter(torch.tensor(1.0))
        self.gamma = torch.nn.Parameter(torch.tensor(1.0))

        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0  # / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        hx = state[0]
        igates = self.gamma * (torch.mm(input, self.weight_ih.t()))
        hgates = self.rho * (torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates

        hy = self.activation(gates)

        return hy, (hy,)


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
        self, input: torch.Tensor, states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        if states is None:
            temp_states: List[Tuple[torch.Tensor]] = []
            batch = input.size(1)
            for layer in self.layers:
                temp_states.append(
                    (
                        torch.zeros(batch, layer.cell.hidden_size, dtype=input.dtype, device=input.device),
                    )
                )

            states = temp_states

        output_states: List[Tuple[torch.Tensor]] = []
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states.append(out_state)
            i += 1
        return output, output_states


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
