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

import itertools
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from sacrebleu import corpus_bleu

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.core.classes import typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils


class EncDecTranslationRNNTBPEModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        self.use_text_data = self.cfg.get("use_text_data", False)
        self.use_audio_data = self.cfg.get("use_audio_data", True)

    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
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
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
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
            if AccessMixin.is_access_enabled():
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            # if (sample_id + 1) % log_every_n_steps == 0:
            #     self.wer.update(encoded, encoded_len, transcript, transcript_len)
            #     _, scores, words = self.wer.compute()
            #     self.wer.reset()
            #     tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            # if (sample_id + 1) % log_every_n_steps == 0:
            #     compute_wer = True
            # else:
            #     compute_wer = False

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
            if AccessMixin.is_access_enabled():
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

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        # RNNT Decoding
        best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded, encoded_len, return_hypotheses=False, partial_hypotheses=None,
        )

        ground_truths = [self.tokenizer.ids_to_text(sent) for sent in transcript.detach().cpu().tolist()]
        translations = best_hyp
        tensorboard_logs.update({'val_translations': translations, 'ground_truths': ground_truths})

        if self.global_rank == 0:
            pass
            # logging.info("-" * 80)
            # logging.info(f"Ground Truth: {ground_truths[0]}")
            # logging.info(f"Translation : {translations[0]}")

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        keys = list(logs.keys())
        for k in keys:
            new_k = k.replace('val_', 'test_')
            logs[new_k] = logs.pop(k)
        return logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}

        sb_score = self._compute_sacrebleu_score(outputs, eval_mode='val')
        tensorboard_logs = {**val_loss_log, 'val_sacreBLEU': sb_score, 'val_neg_sacreBLEU': -sb_score,}
                            # 'wer': wer, 'cer': cer}
        return {**val_loss_log, 'log': tensorboard_logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if not outputs:
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}

        sb_score = self._compute_sacrebleu_score(outputs, eval_mode='test')
        tensorboard_logs = {**test_loss_log, 'test_sacreBLEU': sb_score, 'test_neg_sacreBLEU': -sb_score,}
                            # 'wer': wer, 'cer': cer}
        return {**test_loss_log, 'log': tensorboard_logs}

    def _compute_sacrebleu_score(self, outputs, eval_mode: str = 'val'):
        sb_score = 0.0
        wer = 1.0
        cer = 1.0
        for output in outputs:
            translations = list(itertools.chain(*[x[f'{eval_mode}_translations'] for x in output]))
            ground_truths = list(itertools.chain(*[x['ground_truths'] for x in output]))

            # Gather translations and ground truths from all workers
            tr_and_gt = [None for _ in range(self.world_size)]
            # we also need to drop pairs where ground truth is an empty string
            if self.world_size > 1:
                dist.all_gather_object(
                    tr_and_gt, [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']
                )
            else:
                tr_and_gt[0] = [(t, g) for (t, g) in zip(translations, ground_truths) if g.strip() != '']

            if self.global_rank == 0:
                _translations = []
                _ground_truths = []
                for rank in range(0, self.world_size):
                    _translations += [t for (t, g) in tr_and_gt[rank]]
                    _ground_truths += [g for (t, g) in tr_and_gt[rank]]

                sacre_bleu = corpus_bleu(_translations, [_ground_truths], tokenize="13a")
                sb_score = sacre_bleu.score  # * self.world_size

                wer = word_error_rate(_translations, _ground_truths, use_cer=False)
                cer = word_error_rate(_translations, _ground_truths, use_cer=True)

                # logging.info(f"SB Score : {sb_score}")
                # logging.info(f"WER Score: {wer}")
                # logging.info(f"CER Score: {cer}")
                # # logging.info(f"Sacre Bleu : {sacre_bleu.score}, World size : {self.world_size}")

                for idx, (gt, tr) in enumerate(zip(_ground_truths, _translations)):
                    logging.info(f"{idx + 1:4d}: GT : {gt}")
                    logging.info(f"{idx + 1:4d}: TR : {tr}")
            else:
                sb_score = 0.0
                wer = 0.0
                cer = 0.0

            # self.log(f"{eval_mode}_sacreBLEU", sb_score, sync_dist=True)

        # return sb_score, wer, cer
        return sb_score

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
