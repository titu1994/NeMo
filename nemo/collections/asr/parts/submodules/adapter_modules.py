# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Iterable

import torch
from torch import nn as nn

from nemo.collections.asr.parts.submodules import jasper
from nemo.collections.asr.parts.submodules import subsampling
from nemo.collections.common.parts import adapter_modules
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.utils import logging


class MaskedConvResidualAddAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, adapter: torch.nn.Module, *, module: 'AdapterModuleMixin'):
        # call the normal adapter strategy forward with packed inputs
        outputs = adapter(input)

        # unpack the input tensor
        input_audio_signal, audio_len = input
        input_audio_signal = input_audio_signal[-1]

        # unpack the output tensor
        audio_signal, audio_len = outputs
        audio_signal = audio_signal[-1]

        # print("input abs mean", input_audio_signal.abs().mean(), "output abs mean", audio_signal.abs().mean())

        # add residual connection with input
        audio_signal = input_audio_signal + audio_signal

        logging.info(f"input - adapter abs diff {(audio_signal - input_audio_signal).abs().mean()}")
        # logging.info(f"val ({input_audio_signal[0, 0, :] - audio_signal[0, 0, :]})")

        del input_audio_signal

        # repack the output tensor
        outputs = ([audio_signal], audio_len)
        return outputs


@dataclass
class MaskedConvResidualAddAdapterStrategyConfig:
    _target_: str = "{0}.{1}".format(
        MaskedConvResidualAddAdapterStrategy.__module__, MaskedConvResidualAddAdapterStrategy.__name__
    )  # mandatory field


class MaskedConvAdapter(adapter_modules.AbstractAdapterModule):
    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `post`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
    """

    def __init__(
        self,
        in_features: int,
        repeat: int,
        kernel_size: int = 3,
        activation: str = 'swish',
        dropout: float = 0.0,
        adapter_strategy: MaskedConvResidualAddAdapterStrategy = None,
    ):
        super().__init__()

        activation = activation_registry[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        self.norm_position = 'post'

        # Jasper Block

        # self.module = nn.Sequential(
        #     jasper.JasperBlock(
        #         inplanes=in_features,
        #         planes=in_features,
        #         repeat=repeat,
        #         kernel_size=kernel_size,
        #         dropout=dropout,
        #         activation=activation,
        #         residual=True,
        #         separable=False,
        #         normalization='layer',
        #         conv_mask=True,
        #         stride=[1],
        #         dilation=[1],
        #     ),
        # )

        # Conv2D

        kernel_size = kernel_size[0] if isinstance(kernel_size, Iterable) else kernel_size
        self._kernel_size = kernel_size
        self._padding = jasper.get_same_padding(kernel_size, stride=1, dilation=1)
        self._stride = 1
        self._ceil_mode = False

        in_length = torch.tensor(in_features, dtype=torch.float)
        out_length = subsampling.calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=1,
        )

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=in_features, kernel_size=kernel_size, padding=self._padding, bias=False
            ),
            activation,
        )

        self.out = nn.Sequential(
            torch.nn.Linear(in_features * int(out_length), in_features, bias=False),
            # activation,
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(in_features)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

        # Setup adapter strategy
        if adapter_strategy is None:
            adapter_strategy = MaskedConvResidualAddAdapterStrategyConfig()

        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Final layer initializations must be 0
        self.norm.weight.data *= 0
        self.norm.bias.data *= 0

        # self.module[0].weight.data *= 1e-6
        self.out[0].weight.data *= 0

    # def forward(self, x):
    #     # type: (Tuple[List[torch.Tensor], Optional[torch.Tensor]]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]] # nopep8
    #     x_outputs, length = self.module(x)
    #     x = x_outputs[-1]
    #     x = x.transpose(1, 2)  # [B, T, D]
    #     x = self.norm(x)
    #     x = self.scale * x
    #     x = x.transpose(1, 2)  # [B, D, T]
    #     return [x], length

    def forward(self, x):
        # type: (Tuple[List[torch.Tensor], Optional[torch.Tensor]]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]] # nopep8
        x, x_len = x
        x = x[-1]

        x = x.transpose(1, 2)  # [B, T, D]
        x = x.unsqueeze(1)  # [B, 1, T, D]
        x = self.module(x)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # x = torch.tanh(x) * torch.sigmoid(x)

        # x = self.norm(x)
        # x = self.scale * x
        x = x.transpose(1, 2)  # [B, D, T]
        return [x], x_len


@dataclass
class MaskedConvAdapterConfig:
    in_features: int
    repeat: int
    kernel_size: int = 3
    activation: str = 'swish'
    dropout: float = 0.0
    adapter_strategy: Optional[dict] = MaskedConvResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(MaskedConvAdapter.__module__, MaskedConvAdapter.__name__)
