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

from math import ceil
from typing import Union

import torch

from nemo.core.classes.mixins import DistillationMixin
from nemo.utils import logging


class NMTDistillationMixin(DistillationMixin):
    """
    Distillation Mixin specialization for EncDec based MT models.
    """

    def prehook_additional_distillation_losses(
        self,
        loss_name: str,
        student_registry: Union[list, dict],
        teacher_registry: Union[list, dict],
        teacher_model: 'EncDecMTModel',
    ):
        """
        Pre-hook when computing additional distillation losses. Modifications to the registry here is utilized
        when computing additional losses.

        Note that the pre-hook is called only for the student model. Therefore, `self` refers to the student.

        Args:
            loss_name: str name of the loss function which will be used.
            student_registry: Can be a list or a dictionary.
                When it is a list, items in it represent a tensor that will be paired with the corresponding teacher
                list, and passed to a loss function that accepts binary arguments as its input. Used primarily for
                similarity based losses.
                When it is a dictionary, represents a key-value entry that is merged with between student and teacher.
                Student and teacher should have unique keys, otherwise the merge step will overwrite either the student
                or teacher's values. The dictionary is then passed as a kwarg to the corresponding loss function.
            teacher_registry: Can be a list or a dictionary.
                When it is a list, items in it represent a tensor that will be paired with the corresponding teacher
                list, and passed to a loss function that accepts binary arguments as its input. Used primarily for
                similarity based losses.
                When it is a dictionary, represents a key-value entry that is merged with between student and teacher.
                Student and teacher should have unique keys, otherwise the merge step will overwrite either the student
                or teacher's values. The dictionary is then passed as a kwarg to the corresponding loss function.
            teacher_model: The teacher model in the distillation training. To reference the student model, use `self`.
        """
        if loss_name == 'cosine':
            if self.distill_cfg.get('distill_encoder', False):
                self._distill_additional_loss_mt_encoder(student_registry, teacher_model, teacher_registry)

    def _distill_additional_loss_mt_encoder(self, student_registry, teacher_model, teacher_registry):
        """
        Utility method to perform MTEncoder specific joint loss distillation training.

        For MTEncoder models, will extract the intermediate outputs per Block between teacher and student.
        Then performing near linear mapping between teacher and student to compute cosine embedding loss.

        Args:
            student_registry:
            teacher_model:
            teacher_registry:

        Returns:

        """
        # MTEncoder compatible teacher and student models
        # Get student encoder's registered tensors
        student_encoder_registry = self.get_distillation_module_registry(self.encoder)
        # Get teacher encoder's registered tensors
        teacher_encoder_registry = self.get_distillation_module_registry(teacher_model.encoder)

        # Flatten student registry, extracting just the nested list of tensors registered to cosine loss.
        student_encoder_tensor_list = self.flatten_distillation_module_registry(
            student_encoder_registry, loss_name='cosine'
        )
        # Flatten student registry, extracting just the nested list of tensors registered to cosine loss.
        teacher_encoder_tensor_list = self.flatten_distillation_module_registry(
            teacher_encoder_registry, loss_name='cosine'
        )

        # The above nested lists describe a nested
        # List of blocks (# of jasper blocks) of List of Sub blocks (only last sub block) of tensors
        # Flatten the tensor lists (across the individual sub-modules)
        student_encoder_tensor_list = [mod[0] for mod in student_encoder_tensor_list]  # only last sub-block available
        teacher_encoder_tensor_list = [mod[0] for mod in teacher_encoder_tensor_list]  # only last sub-block available

        # Distribute the teacher layers across the student layers
        num_student_layers = len(student_encoder_tensor_list)  # num student blocks
        num_teacher_layers = len(teacher_encoder_tensor_list)  # num teacher blocks
        stride = num_teacher_layers / float(num_student_layers)

        # for each student block
        for s_idx, student_t in enumerate(student_encoder_tensor_list):
            # find closest teacher block index (clamp at max number of teacher blocks)
            t_idx = max(0, int(ceil(s_idx * stride)))
            t_idx = min(t_idx, num_teacher_layers - 1)

            # select teacher block
            teacher_t = teacher_encoder_tensor_list[t_idx]

            # If student and teacher block shapes match, then perform distillation
            if student_t.shape == teacher_t.shape:
                student_registry.append(student_t)
                teacher_registry.append(teacher_t)
