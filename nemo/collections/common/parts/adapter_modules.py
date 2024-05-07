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

from dataclasses import dataclass, field, is_dataclass
from typing import Any, Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn as nn

from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import access_mixins, adapter_mixin_strategies


class AdapterModuleUtil(access_mixins.AccessMixin):
    """
    Base class of Adapter Modules, providing common functionality to all Adapter Modules.
    """

    def setup_adapter_strategy(self, adapter_strategy: Optional[adapter_mixin_strategies.AbstractAdapterStrategy]):
        """
        Setup adapter strategy of this class, enabling dynamic change in the way the adapter output is
        merged with the input.

        When called successfully, will assign the variable `adapter_strategy` to the module.

        Args:
            adapter_strategy: Can be a None or an implementation of AbstractAdapterStrategy.
        """
        # set default adapter strategy
        if adapter_strategy is None:
            adapter_strategy = self.get_default_strategy_config()

        if is_dataclass(adapter_strategy):
            adapter_strategy = OmegaConf.structured(adapter_strategy)
            OmegaConf.set_struct(adapter_strategy, False)

        # The config must have the `_target_` field pointing to the actual adapter strategy class
        # which will load that strategy dynamically to this module.
        if isinstance(adapter_strategy, dict) or OmegaConf.is_config(adapter_strategy):
            self.adapter_strategy = instantiate(adapter_strategy)
        elif isinstance(adapter_strategy, adapter_mixin_strategies.AbstractAdapterStrategy):
            self.adapter_strategy = adapter_strategy
        else:
            raise AttributeError(f'`adapter_strategy` provided is invalid : {adapter_strategy}')

    def get_default_strategy_config(self) -> 'dataclass':
        """
        Returns a default adapter module strategy.
        """
        return adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()

    def adapter_unfreeze(self,):
        """
        Sets the requires grad for all parameters in the adapter to True.
        This method should be overridden for any custom unfreeze behavior that is required.
        For example, if not all params of the adapter should be unfrozen.
        """
        for param in self.parameters():
            param.requires_grad_(True)


class LinearAdapter(nn.Module, AdapterModuleUtil):

    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `pre`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
        dropout: float value, whether to perform dropout on the output of the last layer of the adapter.
        adapter_strategy: By default, ResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = 'pre',
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()

        activation = activation_registry[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        assert norm_position in ['pre', 'post']
        self.norm_position = norm_position

        if norm_position == 'pre':
            self.module = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
            )

        elif norm_position == 'post':
            self.module = nn.Sequential(
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
                nn.LayerNorm(in_features),
            )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Final layer initializations must be 0
        if self.norm_position == 'pre':
            self.module[-1].weight.data *= 0

        elif self.norm_position == 'post':
            self.module[-1].weight.data *= 0
            self.module[-1].bias.data *= 0

    def forward(self, x):
        x = self.module(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x


@dataclass
class LinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'pre'
    dropout: float = 0.0
    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(LinearAdapter.__module__, LinearAdapter.__name__)


class LoraAdapter(nn.Module, AdapterModuleUtil):

    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `pre`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
        dropout: float value, whether to perform dropout on the output of the last layer of the adapter.
        adapter_strategy: By default, ResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = float(alpha) / float(r)
        self.in_features = in_features
        self.out_features = out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.lora_A.reset_parameters()

        self.lora_B.weight.data *= 0.0
        if self.lora_B.bias is not None:
            self.lora_B.bias.data *= 0.0

    def forward(self, x):
        # Add dropout if available
        x = self.dropout(x)

        x = self.lora_A(x)
        x = self.lora_B(x)
        x = x * self.scaling

        return x


@dataclass
class LoraAdapterConfig:
    in_features: int
    out_features: int
    r: int
    alpha: int
    dropout: float = 0.0

    adapter_strategy: Optional[Any] = field(
        default_factory=lambda: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    )
    _target_: str = "{0}.{1}".format(LoraAdapter.__module__, LoraAdapter.__name__)



def extract_input_output_dims(base_layer: nn.Module):
    # Imported from https://github.com/huggingface/peft/blob/02ae6bcb373d9d9d3bec9ba920d63316418ff64a/src/peft/tuners/lora/layer.py#L92
    if isinstance(base_layer, nn.Linear):
        in_features, out_features = base_layer.in_features, base_layer.out_features
    elif isinstance(base_layer, nn.Conv2d):
        in_features, out_features = base_layer.in_channels, base_layer.out_channels
    elif isinstance(base_layer, nn.Embedding):
        in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
    elif isinstance(base_layer, nn.Conv1d):
        in_features, out_features = (
            base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
        )
    elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
        # QuantLinear
        in_features, out_features = base_layer.infeatures, base_layer.outfeatures
    elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
        # Megatron ColumnParallelLinear,RowParallelLinear
        in_features, out_features = base_layer.input_size, base_layer.output_size
    elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
        # AQLM QuantLinear
        in_features, out_features = base_layer.in_features, base_layer.out_features
    elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
        # Awq layers
        in_features, out_features = base_layer.in_features, base_layer.out_features
    elif base_layer.__class__.__name__ == "EetqLinear":
        # Eetq layers
        in_features, out_features = base_layer.in_features, base_layer.out_features
    elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
        # HQQ layers
        in_features, out_features = base_layer.in_features, base_layer.out_features
    else:
        raise ValueError(f"Unsupported layer type {type(base_layer)}")

    return in_features, out_features