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

from nemo.core.classes.mixins.distill_mixins import DistillationMixin
import torch


class MultiLayerPerceptron(torch.nn.Module, DistillationMixin):
    """
    A simple MLP that can either be used independently or put on top
    of pretrained models (such as BERT) and act as a classifier.
    Args:
        hidden_size (int): the size of each layer
        num_classes (int): number of output classes
        num_layers (int): number of layers
        activation (str): type of activations for layers in between
        log_softmax (bool): whether to add a log_softmax layer before output
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        activation: str = 'relu',
        log_softmax: bool = True,
    ):
        super().__init__()
        self.layers = 0
        for _ in range(num_layers - 1):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            setattr(self, f'layer{self.layers}', layer)
            setattr(self, f'layer{self.layers + 1}', getattr(torch, activation))
            self.layers += 2
        layer = torch.nn.Linear(hidden_size, num_classes)
        setattr(self, f'layer{self.layers}', layer)
        self.layers += 1
        self.log_softmax = log_softmax

    @property
    def last_linear_layer(self):
        return getattr(self, f'layer{self.layers - 1}')

    def forward(self, hidden_states):
        output_states = hidden_states[:]
        for i in range(self.layers):
            output_states = getattr(self, f'layer{i}')(output_states)

        # # Hinton Distillation
        # if self.is_being_distilled():
        #     temp_log_probs = torch.nn.functional.log_softmax(output_states / self.distill_cfg.get('temperature', 1.0))
        #     self.distillation_registration_step(log_prob=temp_log_probs)
        #     del temp_log_probs

        if self.log_softmax:
            output_states = torch.log_softmax(output_states, dim=-1)
        return output_states
