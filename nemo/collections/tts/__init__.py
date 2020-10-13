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

<<<<<<< HEAD
from nemo.collections.tts.data_layers import AudioDataLayer
from nemo.collections.tts.fastspeech_modules import *
from nemo.collections.tts.fastspeech_modules import __all__ as fastspeech__all__
from nemo.collections.tts.parts.helpers import *
from nemo.collections.tts.parts.helpers import __all__ as helpers__all__
from nemo.collections.tts.tacotron2_modules import *
from nemo.collections.tts.tacotron2_modules import __all__ as tacotron2__all__
from nemo.collections.tts.talknet_modules import *
from nemo.collections.tts.talknet_modules import __all__ as fasterspeech__all__
from nemo.collections.tts.waveglow_modules import *
from nemo.collections.tts.waveglow_modules import __all__ as waveglow__all__

__all__ = (
    ["AudioDataLayer"] + helpers__all__ + tacotron2__all__ + waveglow__all__ + fastspeech__all__ + fasterspeech__all__
)
=======
import nemo.collections.tts.data
import nemo.collections.tts.helpers
import nemo.collections.tts.models
>>>>>>> fd98a89adf80012987851a2cd3c3f4dc63bb8db6
