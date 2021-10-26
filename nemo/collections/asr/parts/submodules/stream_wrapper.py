# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn as nn

from enum import Enum

from nemo.utils import logging


class StreamInferenceMode(Enum):
    # Model is in a training state. No streaming is done.
    TRAINING = 'TRAINING'

    # Below are three options for inference:

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is kept inside of the model
    STREAM_INTERNAL_STATE_INFERENCE = 'STREAM_INTERNAL_STATE_INFERENCE'

    # Model is in inference mode and has state for efficient
    # computation/streaming, where state is received from outside of the model
    STREAM_EXTERNAL_STATE_INFERENCE = 'STREAM_EXTERNAL_STATE_INFERENCE'

    # Model its in inference mode and it's topology is the same with training
    # mode (with removed droputs etc)
    NON_STREAM_INFERENCE = 'NON_STREAM_INFERENCE'


class StreamWrapper(nn.Module):
    """
    Streaming wrapper - it is not a standalone layer.
    It can be used to wrap Keras layer for streaming inference mode.
    Advantage of streaming inference mode - it is more computationally efficient.
    But not all layers are streamable. Some layers require keeping a buffer
    with features in time. We can wrap such layer by Stream().
    Where Stream() will create and keep a temporal buffer called state,
    for both cases: internal state and external state.
    Examples of layers which require temporal buffer/state
    for streaming inference are Conv2D, DepthwiseConv2D, AveragePooling2D,
    Flatten in time dimension, etc.

    This wrapper is generic enough, so that it can be used for any modes:
    1 Streaming with internal state. This wrapper will manage internal state.
    2 Streaming with external state. Developer will have to manage external state
    and feed it as additional input to the model and then receive output with
    updated state.
    3 Non streaming inference mode. In this case wrapper will just call
    a wrapped layer as it is. There will be no difference in efficiency.
    The graph will be the same as in training mode, but some training features
    will be removed (such as dropout, etc)
    4 Training mode.

    Attributes:
      module: keras layer which has to be streamed or tf.identity
      inference_batch_size: batch size in inference mode
      mode: inference or training mode
      pad_time_dim: padding in time
      state_shape:
      ring_buffer_size_in_time_dim: size of ring buffer in time dim
      samplewise_inference: True - model will run one sample per one inference step;
        False - model will run multiple per one inference step.
        It is useful for strided streaming

    Raises:
      ValueError: if padding is not 'valid' in streaming mode;
                  or if striding is used with use_one_step;
                  or cell is not supported
    """

    def __init__(
        self,
        module,
        inference_batch_size=1,
        mode=StreamInferenceMode.TRAINING,
        pad_time_dim=None,
        state_shape=None,
        ring_buffer_size_in_time_dim=None,
        samplewise_inference=True,
    ):
        super().__init__()

        self.inner_module = module
        self.inference_batch_size = inference_batch_size
        self.mode = mode
        self.pad_time_dim = pad_time_dim
        self.state_shape = state_shape
        self.ring_buffer_size_in_time_dim = ring_buffer_size_in_time_dim
        self.samplewise_inference = samplewise_inference
        self.stride = 1

        wrapped_module = module

        if not samplewise_inference and isinstance(
            wrapped_module, (nn.Flatten, nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)
        ):
            raise ValueError(
                'Flatten, AdaptiveAvgPool1d, AdaptiveMaxPool1d '
                'can be used only with samplewise_inference = True '
                'because they are executed one time per inference call '
                'and produce only one output in time dim, whereas conv '
                'can produce multiple outputs in time dim, '
                'so conv can be used with samplewise_inference = False or True'
            )

        if self.ring_buffer_size_in_time_dim is not None:
            # it is a special case when ring_buffer_size_in_time_dim is specified
            # outside of the layer in this case we just build a ring buffer
            # and do not check what is the type of the cell
            pass

        elif isinstance(
            wrapped_module,
            (
                nn.Conv1d,
                nn.Conv2d,
                # average_pooling2d.AveragePooling2D),
            ),
        ):
            padding = wrapped_module.padding
            strides = wrapped_module.stride
            self.stride = strides[0]

            if self.mode not in (StreamInferenceMode.TRAINING, StreamInferenceMode.NON_STREAM_INFERENCE):
                # if padding != 'valid':
                #     raise ValueError('conv/cell padding has to be valid,' 'padding has to be set by pad_time_dim')
                if padding != 0:
                    raise ValueError('conv/cell padding has to be 0,' 'padding has to be set by pad_time_dim')

                if self.samplewise_inference:
                    if strides[0] > 1:
                        raise ValueError(
                            'Stride in time dim greater than 1 '
                            'in streaming mode with samplewise_inference=True'
                            ' is not supported, set samplewise_inference=False'
                        )

            dilation_rate = wrapped_module.dilation
            kernel_size = wrapped_module.kernel_size

            if self.samplewise_inference:
                # effective kernel size in time dimension
                self.ring_buffer_size_in_time_dim = dilation_rate[0] * (kernel_size[0] - 1) + 1
            else:
                # Streaming of strided or 1 step conv.
                # Assuming input length is a multiple of strides (otherwise streaming
                # conv is not meaningful), setting to this value (instead of
                # dilation_rate[0] * (kernel_size[0] - 1)) ensures that we do not
                # ignore the `strides - 1` rightmost (and hence most recent) valid
                # input samples.
                self.ring_buffer_size_in_time_dim = max(0, dilation_rate[0] * (kernel_size[0] - 1) - (strides[0] - 1))

        elif isinstance(wrapped_module, nn.AvgPool1d):
            strides = wrapped_module.stride
            kernel_size = wrapped_module.kernel_size
            self.stride = strides[0]

            if (
                self.mode not in (StreamInferenceMode.TRAINING, StreamInferenceMode.NON_STREAM_INFERENCE)
                and strides[0] != kernel_size[0]
            ):
                raise ValueError('Stride in time %d must = pool size in time %d' % (strides[0], kernel_size[0]))
            # effective kernel size in time dimension
            self.ring_buffer_size_in_time_dim = kernel_size[0]

        elif isinstance(
            wrapped_module,
            (nn.Flatten, nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d),
        ):
            # effective kernel size in time dimension
            if self.state_shape:
                self.ring_buffer_size_in_time_dim = self.state_shape[1]

        else:
            raise ValueError('Cell is not supported ', wrapped_module)

        if self.ring_buffer_size_in_time_dim == 1:
            logging.warning('There is no need to use Stream on time dim with size 1')
