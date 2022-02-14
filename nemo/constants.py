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

NEMO_ENV_VARNAME_ENABLE_COLORING = "NEMO_ENABLE_COLORING"
NEMO_ENV_VARNAME_REDIRECT_LOGS_TO_STDERR = "NEMO_REDIRECT_LOGS_TO_STDERR"
NEMO_ENV_VARNAME_TESTING = "NEMO_TESTING"  # Set to True to enable nemo.util.logging's debug mode
NEMO_ENV_VARNAME_VERSION = "NEMO_EXPM_VERSION"  # Used for nemo.utils.exp_manager versioning
NEMO_ENV_CACHE_DIR = "NEMO_CACHE_DIR"  # Used to change default nemo cache directory


import torch
from typing import List, Union


class monitor_cuda_mem:
    _CONTEXT_DEPTH = 0

    def __init__(self, scope, prev=None, empty=False, enabled: bool = True):
        self.scope = scope
        self.prev = prev
        self.empty = empty
        self.enabled = enabled

    def __enter__(self):
        monitor_cuda_mem._CONTEXT_DEPTH += 1

        if self.enabled:
            self.print_pad()
            print(f"|> {self.scope}")

        if self.prev is None:
            self.initial_memory = torch.cuda.memory_allocated(0)
        else:
            self.initial_memory = self.prev
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.empty:
            torch.cuda.empty_cache()

        self.final_memory = torch.cuda.memory_allocated(0)

        if self.enabled:
            self.print_pad()
            print(f"{self.scope} |> {HumanBytes.format(self.final_memory - self.initial_memory)}")

        monitor_cuda_mem._CONTEXT_DEPTH -= 1

    @classmethod
    def print_pad(cls):
        print('\t' * (cls._CONTEXT_DEPTH - 1), end='')


# Shortened form of the answer from https://stackoverflow.com/a/63839503
class HumanBytes:
    # fmt: off
    METRIC_LABELS: List[str] = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    BINARY_LABELS: List[str] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    PRECISION_OFFSETS: List[float] = [0.5, 0.05, 0.005, 0.0005]  # PREDEFINED FOR SPEED.
    PRECISION_FORMATS: List[str] = ["{}{:.0f} {}", "{}{:.1f} {}", "{}{:.2f} {}", "{}{:.3f} {}"]  # PREDEFINED FOR SPEED.
    # fmt: on

    @staticmethod
    def format(num: Union[int, float], metric: bool = False, precision: int = 1) -> str:
        assert isinstance(num, (int, float)), "num must be an int or float"
        assert isinstance(metric, bool), "metric must be a bool"
        assert isinstance(precision, int) and precision >= 0 and precision <= 3, "precision must be an int (range 0-3)"

        unit_labels = HumanBytes.METRIC_LABELS if metric else HumanBytes.BINARY_LABELS
        last_label = unit_labels[-1]
        unit_step = 1000 if metric else 1024
        unit_step_thresh = unit_step - HumanBytes.PRECISION_OFFSETS[precision]

        is_negative = num < 0
        if is_negative:  # Faster than ternary assignment or always running abs().
            num = abs(num)

        for unit in unit_labels:
            if num < unit_step_thresh:
                break
            if unit != last_label:
                num /= unit_step

        return HumanBytes.PRECISION_FORMATS[precision].format("-" if is_negative else "", num, unit)
