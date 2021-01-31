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

import torch
import torch.nn
import copy
import os

import pytest
from omegaconf import DictConfig, ListConfig

from nemo.collections.common.parts import normalization


class TestNormalization:
    @pytest.mark.unit
    def test_broadcast_layer_norm_single_dim(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]
        x2 = x.clone()
        torch_ln = torch.nn.LayerNorm(1024, eps=1e-5)
        broadcast_ln = normalization.BroadcastLayerNorm(1024, dim=[1, None], eps=1e-5)

        # output of pytorch layer norm
        x_t = x.transpose(1, 2)
        orig_ln = torch_ln(x_t)
        orig_ln = orig_ln.transpose(1, 2)

        # output of broadcast layer norm
        brod_ln = broadcast_ln(x2)

        assert orig_ln.shape == brod_ln.shape

        diff = orig_ln - brod_ln
        assert torch.abs(diff).mean() <= 1e-7
        assert torch.square(diff).mean() <= 1e-10

    @pytest.mark.unit
    def test_broadcast_layer_norm_multiple_dim(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]
        x2 = x.clone()
        torch_ln = torch.nn.LayerNorm([1024, 256], eps=1e-5)
        broadcast_ln = normalization.BroadcastLayerNorm([1024, 256], dim=[1, 2], eps=1e-5)

        # output of pytorch layer norm
        orig_ln = torch_ln(x)

        # output of broadcast layer norm
        brod_ln = broadcast_ln(x2)

        assert orig_ln.shape == brod_ln.shape

        diff = orig_ln - brod_ln
        assert torch.abs(diff).mean() <= 5e-6
        assert torch.square(diff).mean() <= 1e-10

    @pytest.mark.unit
    def test_broadcast_layer_norm_single_dim_grad(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]
        x2 = x.clone()
        torch_ln = torch.nn.LayerNorm(1024, eps=1e-5)
        broadcast_ln = normalization.BroadcastLayerNorm(1024, dim=[1, None], eps=1e-5)

        # output of pytorch layer norm
        x_t = x.transpose(1, 2)
        orig_ln = torch_ln(x_t)
        orig_ln = orig_ln.transpose(1, 2)
        orig_ln.sum().backward()

        torch_ln_weight = torch_ln.weight.grad.data
        torch_ln_bias = torch_ln.bias.grad.data

        # output of broadcast layer norm
        brod_ln = broadcast_ln(x2)
        brod_ln.sum().backward()

        brod_ln_weight = broadcast_ln.weight.grad.data
        brod_ln_bias = broadcast_ln.bias.grad.data

        # weights grad check
        assert torch_ln_weight.shape == brod_ln_weight.shape

        diff = torch_ln_weight - brod_ln_weight
        assert torch.abs(diff).mean() <= 1e-4
        assert torch.square(diff).mean() <= 1e-7

        # bias grad check
        assert torch_ln_bias.shape == brod_ln_bias.shape

        diff = torch_ln_bias - brod_ln_bias
        assert torch.abs(diff).mean() <= 1e-7
        assert torch.square(diff).mean() <= 1e-10

    @pytest.mark.unit
    def test_broadcast_layer_norm_multi_dim_grad(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]
        x2 = x.clone()
        torch_ln = torch.nn.LayerNorm([1024, 256], eps=1e-5)
        broadcast_ln = normalization.BroadcastLayerNorm([1024, 256], dim=[1, 2], eps=1e-5)

        # output of pytorch layer norm
        orig_ln = torch_ln(x)
        orig_ln.sum().backward()

        torch_ln_weight = torch_ln.weight.grad.data
        torch_ln_bias = torch_ln.bias.grad.data

        # output of broadcast layer norm
        brod_ln = broadcast_ln(x2)
        brod_ln.sum().backward()

        brod_ln_weight = broadcast_ln.weight.grad.data
        brod_ln_bias = broadcast_ln.bias.grad.data

        # weights grad check
        assert torch_ln_weight.shape == brod_ln_weight.shape

        diff = torch_ln_weight - brod_ln_weight
        assert torch.abs(diff).mean() <= 1e-4
        assert torch.square(diff).mean() <= 1e-7

        # bias grad check
        assert torch_ln_bias.shape == brod_ln_bias.shape

        diff = torch_ln_bias - brod_ln_bias
        assert torch.abs(diff).mean() <= 1e-7
        assert torch.square(diff).mean() <= 1e-10

    @pytest.mark.unit
    def test_broadcast_layer_norm_multiple_dim_invalid_dims(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]

        with pytest.raises(ValueError):
            broadcast_ln = normalization.BroadcastLayerNorm([1024, 256], dim=[1, None], eps=1e-5)

    @pytest.mark.unit
    def test_broadcast_layer_norm_last_dim(self):
        x = torch.randn(32, 1024, 256)  # [B, C, T]
        x2 = x.clone()
        torch_ln = torch.nn.LayerNorm(256, eps=1e-5)
        broadcast_ln = normalization.BroadcastLayerNorm(256, dim=[None, 1], eps=1e-5)

        # output of pytorch layer norm
        orig_ln = torch_ln(x)

        # output of broadcast layer norm
        brod_ln = broadcast_ln(x2)

        assert orig_ln.shape == brod_ln.shape

        diff = orig_ln - brod_ln
        assert torch.abs(diff).mean() <= 1e-7
        assert torch.square(diff).mean() <= 1e-10
