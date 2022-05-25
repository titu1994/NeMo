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

import pytest
import torch

from nemo.collections.asr.parts.submodules import adapter_modules
from nemo.core import NeuralModule
from nemo.core.classes.mixins import AdapterModuleMixin, adapter_mixins
from nemo.utils import config_utils


class DefaultModule(NeuralModule):
    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Linear(50, 50)
        self.bn = torch.nn.BatchNorm1d(50)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x
        return out

    def num_params(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


class DefaultModuleAdapter(DefaultModule, AdapterModuleMixin):
    def forward(self, x):
        x = super(DefaultModuleAdapter, self).forward(x)

        if self.is_adapter_available():
            # For testing purposes, cache the adapter names
            self._adapter_names = self.get_enabled_adapters()
            # call forward over model adapters, summing them up
            x = self.forward_enabled_adapters(x)

        return x


def get_adapter_cfg(in_features=80, repeat=2, kernel_size=3):
    cfg = adapter_modules.MaskedConvAdapterConfig(in_features, repeat, kernel_size=kernel_size)
    return cfg


def get_classpath(cls):
    return f'{cls.__module__}.{cls.__name__}'


if adapter_mixins.get_registered_adapter(DefaultModule) is None:
    adapter_mixins.register_adapter(DefaultModule, DefaultModuleAdapter)


class TestAdapterStrategy:
    @pytest.mark.unit
    def test_MaskedConvResidualAddAdapterStrategyConfig(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.MaskedConvResidualAddAdapterStrategy,
            adapter_modules.MaskedConvResidualAddAdapterStrategyConfig,
            ignore_args=IGNORED_ARGS,
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_MaskedConvResidualAddAdapterStrategy_default(self):
        torch.random.manual_seed(0)
        seq = torch.randn(2, 80, 31)  # (B, D, T)
        seq_len = torch.tensor([25, 31], dtype=torch.int32)
        x = ([seq], seq_len)

        module = DefaultModuleAdapter()
        module.add_adapter(name='temp', cfg=get_adapter_cfg())
        adapter = module.adapter_layer[module.get_enabled_adapters()[0]]

        # update the strategy
        adapter_strategy = adapter_modules.MaskedConvResidualAddAdapterStrategy()
        adapter.adapter_strategy = adapter_strategy

        with torch.no_grad():
            out, _ = adapter_strategy.forward(x, adapter, module=module)
            out = out[-1]
            assert (out - seq).abs().mean() < 1e-5
