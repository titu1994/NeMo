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
import numbers
from typing import Union, List, Optional


_shape_t = Union[int, List[int], torch.Size]


class BroadcastLayerNorm(torch.nn.LayerNorm):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    dim: Union[int, List[int]] = None
    eps: float
    elementwise_affine: bool

    def __init__(
        self, normalized_shape: _shape_t, dim: List[Optional[int]], eps: float = 1e-5, elementwise_affine: bool = True,
    ):
        r"""Applies Layer Normalization over a mini-batch of inputs as described in
        the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        The mean and standard-deviation are calculated separately over the last
        certain number dimensions which have to be of the shape specified by
        :attr:`normalized_shape`.
        :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
        :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
        The standard-deviation is calculated via the biased estimator, equivalent to
        `torch.var(input, unbiased=False)`.

        .. note::
            Unlike Batch Normalization and Instance Normalization, which applies
            scalar scale and bias for each entire channel/plane with the
            :attr:`affine` option, Layer Normalization applies per-element scale and
            bias with :attr:`elementwise_affine`.

        This layer uses statistics computed from input data in both training and
        evaluation modes.

        Args:
            normalized_shape (int or list or torch.Size): input shape from an expected input
                of size

                .. math::
                    [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                        \times \ldots \times \text{normalized\_shape}[-1]]

                If a single integer is used, it is treated as a singleton list, and this module will
                normalize over the last dimension which is expected to be of that specific size.
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: a boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.

        Shape:
            - Input: :math:`(N, *)`
            - Output: :math:`(N, *)` (same shape as input)

        Examples::

            >>> input = torch.randn(20, 5, 10, 10)
            >>> # With Learnable Parameters
            >>> m = nn.LayerNorm(input.size()[1:])
            >>> # Without Learnable Parameters
            >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
            >>> # Normalize over last two dimensions
            >>> m = nn.LayerNorm([10, 10])
            >>> # Normalize over last dimension of size 10
            >>> m = nn.LayerNorm(10)
            >>> # Activating the module
            >>> output = m(input)
        """
        super(BroadcastLayerNorm, self).__init__(
            normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )
        if dim is None:
            dim = [-1]

        assert type(dim) in (tuple, list)
        assert len(dim) >= 1
        for d in dim:
            if d is not None and d < 1:
                raise ValueError("dim must be indices other than batch dimension.")

        dix = 0
        broadcast_dim = []
        for d in dim:
            if d is not None:
                broadcast_dim.append(self.normalized_shape[dix])
                dix += 1
            else:
                broadcast_dim.append(1)

        if dix != len(self.normalized_shape):
            raise ValueError("number of non-None dims provided is not equal to rank of `normalized_shape` provided!")

        self.dim = [d for d in dim if d is not None]
        self._broadcast_mask = [1] + broadcast_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        var, mean = torch.var_mean(x, dim=self.dim, unbiased=False, keepdim=True)

        if (
            not torch.is_tensor(self.eps)
            or self.eps.device != self.weight.device
            or self.eps.dtype != self.weight.dtype
        ):
            self.eps = torch.tensor(self.eps, dtype=self.weight.dtype, device=self.weight.device)

        std = (var + self.eps).sqrt()
        y = (x - mean) / std

        if self.weight is not None:
            print(self._broadcast_mask)
            y *= self.weight.view(self._broadcast_mask)

        if self.bias is not None:
            y += self.bias.view(self._broadcast_mask)
        return y

    def extra_repr(self) -> torch.Tensor:
        extra_repr = super(BroadcastLayerNorm, self).extra_repr()
        extra_repr = extra_repr + ", dim={dim}".format(**self.__dict__)
        return extra_repr
