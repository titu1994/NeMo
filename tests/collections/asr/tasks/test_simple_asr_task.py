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
import copy

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data import audio_to_text
from nemo.collections.asr.models.tasks.rnnt_bpe_peft_model import SpeechTaskPEFTModel


@pytest.fixture()
def asr_model():
    modelConfig = DictConfig(
        {'base_model_name': 'stt_en_conformer_ctc_small', 'use_lora': True,
         'model_dim': 256, 'out_dim': 1}
    )

    model_instance = SpeechTaskPEFTModel(cfg=modelConfig).cpu()
    return model_instance


class TestEncDecCTCModel:
    @pytest.mark.unit
    def test_constructor(self, asr_model):
        asr_model.train()
        # TODO: make proper config and assert correct number of weights
        # Check to/from config_dict:
        confdict = asr_model.to_config_dict()
        # instance2 = SpeechTaskPEFTModel.from_config_dict(confdict)
        # assert isinstance(instance2, SpeechTaskPEFTModel)
        print(asr_model.summarize())

    @pytest.mark.unit
    def test_forward(self, asr_model):
        asr_model = asr_model.eval()

        input_signal = torch.randn(size=(4, 512))
        length = torch.randint(low=161, high=500, size=[4])

        with torch.no_grad():
            # batch size 1
            logprobs_instance = []
            for i in range(input_signal.size(0)):
                encoded = asr_model.forward(
                    input_signal=input_signal[i : i + 1], input_signal_length=length[i : i + 1]
                )
                encoded, _ = encoded
                logprobs_instance.append(encoded)
            logprobs_instance = torch.cat(logprobs_instance, 0)

            # batch size 4
            logprobs_batch, _ = asr_model.forward(input_signal=input_signal, input_signal_length=length)

        assert logprobs_instance.shape == logprobs_batch.shape
        diff = torch.mean(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 3e-6
        diff = torch.max(torch.abs(logprobs_instance - logprobs_batch))
        assert diff <= 3e-6
