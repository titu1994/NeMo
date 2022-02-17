# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import argparse

from nemo.core import ModelPT
from nemo.utils import model_utils

DOMAINS = ['asr', 'tts', 'nlp']


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', choices=DOMAINS, type=str)

    args = parser.parse_args()
    return args


###############################


def _build_import_path(domain, imp):
    path = ".".join(["nemo", "collections", domain, "models", imp])
    return path


def _get_class_from_path(domain, imp):
    path = _build_import_path(domain, imp)
    class_ = model_utils.import_class_by_path(path)

    result = None
    if inspect.isclass(class_) and issubclass(class_, ModelPT):
        result = class_

    return result


###############################


def test_domain_asr(args):
    import nemo.collections.asr as nemo_asr

    module_list = []
    for imp in dir(nemo_asr.models):
        class_ = _get_class_from_path(args.domain, imp)
        if class_ is not None:
            module_list.append(class_)

    for module in module_list:
        print("Module successfully imported :", module)


def test_domain_nlp(args):
    import nemo.collections.nlp as nemo_nlp

    module_list = []
    for imp in dir(nemo_nlp.models):
        class_ = _get_class_from_path(args.domain, imp)
        if class_ is not None:
            module_list.append(class_)

    for module in module_list:
        print("Module successfully imported :", module)


def test_domain_tts(args):
    import nemo.collections.tts as nemo_tts

    module_list = []
    for imp in dir(nemo_tts.models):
        class_ = _get_class_from_path(args.domain, imp)
        if class_ is not None:
            module_list.append(class_)

    for module in module_list:
        print("Module successfully imported :", module)


###############################


def test_domain(args):
    domain = args.domain

    if domain == 'asr':
        test_domain_asr(args)
    elif domain == 'nlp':
        test_domain_nlp(args)
    elif domain == 'tts':
        test_domain_tts(args)
    else:
        raise RuntimeError(f"Cannot resolve domain : {domain}")


def run_checks():
    args = process_args()
    test_domain(args)


if __name__ == '__main__':
    run_checks()
