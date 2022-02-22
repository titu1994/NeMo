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

import time


class monitor_cuda_mem:
    _CONTEXT_DEPTH = 0
    ENABLED = True
    EMPTY = False

    def __init__(self, scope, empty=None, enabled: bool = False, precision: int = 4):
        self.scope = scope
        self.empty = empty if empty is not None else monitor_cuda_mem.EMPTY
        self.enabled = enabled if enabled is not None else monitor_cuda_mem.ENABLED
        self.precision = precision

    def __enter__(self):
        monitor_cuda_mem._CONTEXT_DEPTH += 1

        if self.enabled:
            self.print_pad()
            print(f"|> {self.scope}")

            self.initial_memory = torch.cuda.memory_allocated(0)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if self.empty:
                torch.cuda.empty_cache()

            self.final_memory = torch.cuda.memory_allocated(0)

            memory_diff = HumanBytes.format(self.final_memory - self.initial_memory, precision=self.precision)
            self.print_pad()
            print(f"{self.scope} |> {memory_diff}")

        monitor_cuda_mem._CONTEXT_DEPTH -= 1

    @classmethod
    def print_pad(cls):
        print('\t' * (cls._CONTEXT_DEPTH - 1), end='')


# Shortened form of the answer from https://stackoverflow.com/a/63839503
class HumanBytes:
    # fmt: off
    METRIC_LABELS: List[str] = ["B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    BINARY_LABELS: List[str] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    PRECISION_OFFSETS: List[float] = [5 * (0.1 ** x) for x in range(1, 22)]  # PREDEFINED FOR SPEED.
    PRECISION_FORMATS: List[str] = [("{}{:." + str(ratio) + "f} {}") for ratio in range(len(PRECISION_OFFSETS))]  # PREDEFINED FOR SPEED.

    # fmt: on

    @staticmethod
    def format(num: Union[int, float], metric: bool = False, precision: int = 1) -> str:
        assert isinstance(num, (int, float)), "num must be an int or float"
        assert isinstance(metric, bool), "metric must be a bool"
        assert isinstance(precision, int) and precision >= 0 and precision <= len(HumanBytes.PRECISION_OFFSETS), "precision must be an int (range 0-20)"

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


class monitor_time:
    _CONTEXT_DEPTH = 0
    ENABLED = True
    CUDA_SYNC = True

    def __init__(self, scope: str, enabled: bool = False, cuda_sync: bool = None):
        self.scope = scope
        self.enabled = enabled if enabled is not None else monitor_time.ENABLED
        self.cuda_sync = cuda_sync if cuda_sync is not None else monitor_time.CUDA_SYNC

    def __enter__(self):
        monitor_time._CONTEXT_DEPTH += 1

        if self.enabled:
            self.print_pad()
            print(f"|> {self.scope}")

            self.initial_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if self.cuda_sync:
                torch.cuda.synchronize()

            self.final_time = time.time()

            self.print_pad()
            print(f"{self.scope} |> {(self.final_time - self.initial_time)}",)

        monitor_time._CONTEXT_DEPTH -= 1

    @classmethod
    def print_pad(cls):
        print('\t' * (cls._CONTEXT_DEPTH - 1), end='')
