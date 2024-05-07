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
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import _AudioTextDataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTModel, EncDecRNNTBPEModel
from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)
from nemo.collections.common.parts.adapter_modules import LoraAdapter, LoraAdapterConfig, extract_input_output_dims
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.mixins import AccessMixin, AdapterModelPTMixin
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins.adapter_mixins import freeze_batchnorm_modules
from nemo.utils import logging, model_utils


class SpeechTaskPEFTModel(ASRModel, ASRModuleMixin):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # # Convert to Hydra 1.0 compatible DictConfig
        # cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # cfg = model_utils.maybe_update_config_version(cfg)
        #
        # if not isinstance(cfg, DictConfig):
        #     cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        self.base_model: Optional[Union[ASRModel, AdapterModelPTMixin]] = None
        if cfg.get("base_model") is not None:
            # load directly from config
            # either if config provided initially, or automatically
            # after model restoration
            target_class = model_utils.import_class_by_path(self.cfg.base_model.target)

            self.register_nemo_submodule(
                name="base_model", config_field="base_model", model=target_class(self.cfg.base_model, trainer=trainer),
            )
        elif cfg.get('base_model_path') is not None:
            # load from .nemo model checkpoint
            # while saving, config will be automatically assigned/updated
            # in cfg.child_model
            self.register_nemo_submodule(
                name="base_model",
                config_field="base_model",
                model=ASRModel.restore_from(self.cfg.base_model_path, trainer=trainer),
            )
        elif cfg.get('base_model_name') is not None:
            # load from pretrained model
            # while saving, config will be automatically assigned/updated
            # in cfg.child_model
            self.register_nemo_submodule(
                name="base_model",
                config_field="base_model",
                model=ASRModel.from_pretrained(self.cfg.base_model_name, trainer=trainer),
            )

        # Assert type of encoder
        assert isinstance(self.base_model.encoder, ConformerEncoder), "Only ConformerEncoder is supported as encoder"

        # Cleanup unused modules
        del self.base_model.decoder
        if hasattr(self.base_model, 'joint'):
            del self.base_model.joint
        del self.base_model.loss
        del self.base_model.wer
        del self.base_model.decoding

        # Freeze base model
        if self.cfg.get('freeze_base_model', True):
            self.base_model.freeze()

        # Add adapters (if config enables)
        self.use_lora = self.cfg.get("use_lora", True)
        self.lora_cfg = self.cfg.get("lora_cfg", {})
        self.lora_target_modules = self.lora_cfg.get("target_modules", ['linear_q', 'linear_v'])

        if self.use_lora:
            for name, module in self.base_model.named_modules():
                for lora_target in self.lora_target_modules:
                    if lora_target in name:
                        if 'lora_adapter' in name:
                            continue
                        print(f"Adding LoraAdapter to {name}")

                        # freeze batch norm if any in the adapter submodules
                        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            module.track_running_stats = (
                                False  # prevent running stats from updated during finetuning
                            )

                        input_dim, output_dim = extract_input_output_dims(module)

                        lora_cfg = LoraAdapterConfig(
                            in_features=input_dim,
                            out_features=output_dim,
                            r=self.lora_cfg.get("r", 8),
                            alpha=self.lora_cfg.get("alpha", 8),
                            dropout=self.lora_cfg.get("dropout", 0.0),
                        )

                        # Add adapter to module
                        module.lora_adapter = self.from_config_dict(OmegaConf.structured(lora_cfg))

                        # # Update forward pass of module
                        class ModuleWrapper(module.__class__):
                            def forward(self, *args, **kwargs):
                                # Get output of original module
                                module_output = super().forward(*args, **kwargs)

                                if len(args) > 0:
                                    ip = args[0]
                                else:
                                    ip = kwargs[list(kwargs.keys())[0]]

                                adapter_output = self.lora_adapter(ip)
                                return module_output + adapter_output

                        # Update module class
                        module.__class__ = ModuleWrapper

        # Add new modules
        self.output = torch.nn.Linear(self.cfg.model_dim, self.cfg.out_dim, bias=False)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        # if config.get("use_lhotse"):
        #     return get_lhotse_dataloader_from_config(
        #         config,
        #         global_rank=self.global_rank,
        #         world_size=self.world_size,
        #         dataset=LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer,),
        #     )
        #
        # dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
        #     config=config,
        #     local_rank=self.local_rank,
        #     global_rank=self.global_rank,
        #     world_size=self.world_size,
        #     tokenizer=self.tokenizer,
        #     preprocessor_cfg=self.cfg.get("preprocessor", None),
        # )
        #
        # if dataset is None:
        #     return None
        #
        # if isinstance(dataset, AudioToBPEDALIDataset):
        #     # DALI Dataset implements dataloader interface
        #     return dataset
        #
        # shuffle = config['shuffle']
        # if isinstance(dataset, torch.utils.data.IterableDataset):
        #     shuffle = False
        #
        # if hasattr(dataset, 'collate_fn'):
        #     collate_fn = dataset.collate_fn
        # elif hasattr(dataset.datasets[0], 'collate_fn'):
        #     # support datasets that are lists of entries
        #     collate_fn = dataset.datasets[0].collate_fn
        # else:
        #     # support datasets that are lists of lists
        #     collate_fn = dataset.datasets[0].datasets[0].collate_fn
        #
        # batch_sampler = None
        # if config.get('use_semi_sorted_batching', False):
        #     if not isinstance(dataset, _AudioTextDataset):
        #         raise RuntimeError(
        #             "Semi Sorted Batch sampler can be used with AudioToCharDataset or AudioToBPEDataset "
        #             f"but found dataset of type {type(dataset)}"
        #         )
        #     # set batch_size and batch_sampler to None to disable automatic batching
        #     batch_sampler = get_semi_sorted_batch_sampler(self, dataset, config)
        #     config['batch_size'] = None
        #     config['drop_last'] = False
        #     shuffle = False
        #
        # return torch.utils.data.DataLoader(
        #     dataset=dataset,
        #     batch_size=config['batch_size'],
        #     sampler=batch_sampler,
        #     batch_sampler=None,
        #     collate_fn=collate_fn,
        #     drop_last=config.get('drop_last', False),
        #     shuffle=shuffle,
        #     num_workers=config.get('num_workers', 0),
        #     pin_memory=config.get('pin_memory', False),
        # )

        return None

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.

        # if (
        #     self._train_dl is not None
        #     and hasattr(self._train_dl, 'dataset')
        #     and isinstance(self._train_dl.dataset, torch.utils.data.IterableDataset)
        # ):
        #     # We also need to check if limit_train_batches is already set.
        #     # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
        #     # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
        #     if self._trainer is not None and isinstance(self._trainer.limit_train_batches, float):
        #         self._trainer.limit_train_batches = int(
        #             self._trainer.limit_train_batches
        #             * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
        #         )
        #     elif self._trainer is None:
        #         logging.warning(
        #             "Model Trainer was not set before constructing the dataset, incorrect number of "
        #             "training batches will be used. Please set the trainer and rebuild the dataset."
        #         )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     if hasattr(self.preprocessor, '_sample_rate'):
    #         input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
    #     else:
    #         input_signal_eltype = AudioSignal()
    #
    #     return {
    #         "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
    #         "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
    #         "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
    #         "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
    #     }
    #
    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
    #         "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
    #     }

    # @typecheck()
    def forward(self, input_signal=None, input_signal_length=None):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """

        processed_signal, processed_signal_length = self.base_model.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        # Spec augment is not applied during evaluation/testing
        if self.base_model.spec_augmentation is not None and self.training:
            processed_signal = self.base_model.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        encoded, encoded_len = self.base_model.encoder(audio_signal=processed_signal, length=processed_signal_length)

        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value}

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     signal, signal_len, transcript, transcript_len, sample_id = batch
    #
    #     # forward() only performs encoder forward
    #     # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
    #     #     encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
    #     # else:
    #     encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
    #     del signal
    #
    #     best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
    #         encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
    #     )
    #
    #     sample_id = sample_id.cpu().detach().numpy()
    #     return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        # if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
        #     encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        # else:
        encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )

                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )

            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'val_wer': wer_num.float() / wer_denom}
        return {'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {'test_wer': wer_num.float() / wer_denom}
        return {'log': tensorboard_logs}
