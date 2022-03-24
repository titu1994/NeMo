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
#
import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadLinearAttention,
    CachedMultiHeadAttention,
    CachedMultiHeadLinearAttention,
    CachedRelPositionMultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.activations import Swish

from nemo.constants import monitor_cuda_mem, monitor_time

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer']


class ConformerLayer(torch.nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        self_attention_type='global',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        global_pos_emb: bool = False,
        pos_bias_u=None,
        pos_bias_v=None,
        shared_attention: bool = False,
        d_ff_bottleneck: int = -1,
        untie_pos_emb: bool = False,
    ):
        super(ConformerLayer, self).__init__()

        self.self_attention_model = self_attention_model
        self.self_attention_type = self_attention_type
        self.n_heads = n_heads
        self.fc_factor = 0.5
        self.shared_attention = shared_attention
        self.global_pos_emb = global_pos_emb
        self.untie_pos_emb = untie_pos_emb

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)

        if d_ff_bottleneck < 0:
            self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        else:
            self.feed_forward1 = ConformerFeedForwardBottleneck(
                d_model=d_model, d_ff=d_ff, d_bottleneck=d_ff_bottleneck, dropout=dropout
            )

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)

        elif self_attention_model is None:
            if self.self_attention_type == 'global':
                if self.shared_attention:
                    self.self_attn = CachedMultiHeadAttention(
                        n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, untie_pos_emb=untie_pos_emb
                    )
                else:
                    self.self_attn = MultiHeadAttention(
                        n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, untie_pos_emb=untie_pos_emb
                    )

            elif self.self_attention_type == 'rel_global':
                if self.shared_attention:
                    self.self_attn = CachedRelPositionMultiHeadAttention(
                        n_head=n_heads,
                        n_feat=d_model,
                        dropout_rate=dropout_att,
                        pos_bias_u=pos_bias_u,
                        pos_bias_v=pos_bias_v,
                        untie_pos_emb=untie_pos_emb,
                    )
                else:
                    self.self_attn = RelPositionMultiHeadAttention(
                        n_head=n_heads,
                        n_feat=d_model,
                        dropout_rate=dropout_att,
                        pos_bias_u=pos_bias_u,
                        pos_bias_v=pos_bias_v,
                        untie_pos_emb=untie_pos_emb,
                    )

            elif self.self_attention_type == 'linear':
                if self.shared_attention:
                    self.self_attn = CachedMultiHeadLinearAttention(
                        n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att
                    )
                else:
                    self.self_attn = MultiHeadLinearAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)

            else:
                raise ValueError("Incorrect value of `self_attention_type`")
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)

        if d_ff_bottleneck < 0:
            self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        else:
            self.feed_forward2 = ConformerFeedForwardBottleneck(
                d_model=d_model, d_ff=d_ff, d_bottleneck=d_ff_bottleneck, dropout=dropout
            )

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, lengths, att_mask=None, pos_emb=None, pad_mask=None, att_cache=None):

        if self.self_attention_model in ['rel_pos', 'abs_pos']:
            return self.forward_with_positional_embeddings(x, lengths, att_mask, pos_emb, pad_mask)

        else:
            if not self.shared_attention:
                if att_cache is not None:
                    del att_cache
                    att_cache = None

            return self.forward_without_positional_embeddings(x, lengths, att_mask, pos_emb, pad_mask, att_cache)

    def forward_with_positional_embeddings(self, x, lengths, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        return x

    def forward_without_positional_embeddings(
        self, x, lengths, att_mask=None, pos_emb=None, pad_mask=None, att_cache=None
    ):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        dtype = x.dtype
        residual = x
        with monitor_cuda_mem('first feed forward'), monitor_time('first feed forward'):
            x = self.norm_feed_forward1(x)
            x = self.feed_forward1(x)
            residual = residual + self.dropout(x) * self.fc_factor

        with monitor_cuda_mem(f'attention (shared={self.shared_attention})'), monitor_time(
            f'attention (shared={self.shared_attention})'
        ):
            x = self.norm_self_att(residual)

            if self.self_attention_type in ('global', 'rel_global'):
                _mask = att_mask
            elif self.self_attention_type == 'linear':
                _mask = pad_mask
            else:
                raise ValueError()

            if self.global_pos_emb or self.untie_pos_emb or self.self_attention_type == 'rel_global':
                _pos = pos_emb
            else:
                _pos = None

            if self.shared_attention:
                x = self.self_attn(x, attention=att_cache, mask=_mask, pos_emb=_pos)
            else:
                x, att_cache = self.self_attn(query=x, key=x, value=x, mask=_mask, pos_emb=_pos, return_attention=True)
            residual = residual + self.dropout(x)

        with monitor_cuda_mem('conv'), monitor_time('conv'):
            x = self.norm_conv(residual)
            x = self.conv(x, pad_mask)
            residual = residual + self.dropout(x)

        with monitor_cuda_mem('second feed forward'), monitor_time('second feed forward'):
            x = self.norm_feed_forward2(residual)
            x = self.feed_forward2(x)
            residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)
        return x.to(dtype=dtype), att_cache


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(self, d_model, kernel_size, norm_type='batch_norm'):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerFeedForwardBottleneck(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, d_bottleneck, dropout, activation=Swish()):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(d_model, d_bottleneck), nn.Linear(d_bottleneck, d_ff),)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Sequential(nn.Linear(d_ff, d_bottleneck), nn.Linear(d_bottleneck, d_model))

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
