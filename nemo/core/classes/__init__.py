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


from nemo.core.classes.common import (
    FileIO,
    Model,
    PretrainedModelInfo,
    Serialization,
    Typing,
    is_typecheck_enabled,
    typecheck,
)
from nemo.core.classes.dataset import Dataset, IterableDataset
from nemo.core.classes.exportable import Exportable, ExportFormat
from nemo.core.classes.loss import Loss
from nemo.core.classes.module import NeuralModule
from nemo.utils import exceptions

# TODO @blisc: Perhaps refactor instead of import guarding
try:
    import pytorch_lightning
    import hydra
    import omegaconf
    from nemo.core.classes.modelPT import ModelPT
except ModuleNotFoundError:
    from nemo.utils.exceptions import CheckInstall

    class ModelPT(CheckInstall):
        pass

    class DistillationModelPT(CheckInstall):
        pass
