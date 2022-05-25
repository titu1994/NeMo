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
from nemo.core.classes.mixins import adapter_mixin_strategies
from nemo.utils import config_utils


class TestASRAdapterModules:
    def check_assertion(self, adapter, input):
        with torch.no_grad():
            assert adapter.norm.weight.sum() == 0
            if hasattr(adapter.norm, 'bias') and adapter.norm.bias is not None:
                assert adapter.norm.bias.sum() == 0

            out, seq_len = adapter(input)
            out = out[-1]
            assert out.mean() <= 1e-8

    @pytest.mark.unit
    def test_masked_conv_adapter_config(self):
        IGNORED_ARGS = ['_target_']

        result = config_utils.assert_dataclass_signature_match(
            adapter_modules.MaskedConvAdapter, adapter_modules.MaskedConvAdapterConfig, ignore_args=IGNORED_ARGS
        )

        signatures_match, cls_subset, dataclass_subset = result

        assert signatures_match
        assert cls_subset is None
        assert dataclass_subset is None

    @pytest.mark.unit
    def test_masked_conv_adapter_init(self):
        torch.random.manual_seed(0)
        seq = torch.randn(2, 80, 31)  # (B, D, T)
        seq_len = torch.tensor([25, 31], dtype=torch.int32)
        x = ([seq], seq_len)

        adapter = adapter_modules.MaskedConvAdapter(in_features=80, repeat=2)

        self.check_assertion(adapter, x)

    @pytest.mark.unit
    @pytest.mark.parametrize('repeat', [1, 2, 3])
    def test_masked_conv_adapter_repeat(self, repeat):
        torch.random.manual_seed(0)
        seq = torch.randn(2, 80, 31)  # (B, D, T)
        seq_len = torch.tensor([25, 31], dtype=torch.int32)
        x = ([seq], seq_len)

        adapter = adapter_modules.MaskedConvAdapter(in_features=80, repeat=repeat)

        self.check_assertion(adapter, x)

    @pytest.mark.unit
    @pytest.mark.parametrize('kernel_size', [1, 2, 3])
    def test_masked_conv_adapter_kernel_size(self, kernel_size):
        torch.random.manual_seed(0)
        seq = torch.randn(2, 80, 31)  # (B, D, T)
        seq_len = torch.tensor([25, 31], dtype=torch.int32)
        x = ([seq], seq_len)

        adapter = adapter_modules.MaskedConvAdapter(in_features=80, repeat=1, kernel_size=kernel_size)

        self.check_assertion(adapter, x)

    @pytest.mark.unit
    def test_masked_conv_adapter_dropout(self):
        torch.random.manual_seed(0)
        seq = torch.randn(2, 80, 31)  # (B, D, T)
        seq_len = torch.tensor([25, 31], dtype=torch.int32)
        x = ([seq], seq_len)

        adapter = adapter_modules.MaskedConvAdapter(in_features=80, repeat=2, dropout=0.5)

        self.check_assertion(adapter, x)

    @pytest.mark.unit
    def test_masked_conv_adapter_strategy(self):
        adapter = adapter_modules.MaskedConvAdapter(in_features=50, repeat=1)
        assert hasattr(adapter, 'adapter_strategy')
        assert adapter.adapter_strategy is not None
        # assert default strategy is set
        assert isinstance(adapter.adapter_strategy, adapter_modules.MaskedConvResidualAddAdapterStrategy)
