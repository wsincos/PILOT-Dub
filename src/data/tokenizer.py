# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py
# Copyright    2023                            (authors: Feiteng Li)
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

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import torch
import torchaudio
# from lhotse.features import FeatureExtractor
# from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

from src.modeling.modules.encodec import model_from_checkpoint


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        phonemizer = EspeakBackend(
            language,
            punctuation_marks=punctuation_marks, #定义哪些标点符号会被保留并作为音素处理的一部分
            preserve_punctuation=preserve_punctuation, # 是否在生成音素时保留标点符号
            with_stress=with_stress, # 是否在生成的音素中包含重音标记
            tie=tie, # 是否在双音素之间添加连接符号
            language_switch=language_switch, # 指定在多语言文本中如何处理语言切换
            words_mismatch=words_mismatch, # 指定在处理未知单词或无法匹配的单词时的行为
        )
        
        self.backend = phonemizer
        self.separator = separator # 定义分隔符

    def to_list(self, phonemized: str) -> List[str]:
        """
        Convert a phonemized string into a list of phonemes and separators.

        Args:
            phonemized (str): The phonemized string, where words, syllables, and phonemes
                             are separated by predefined separators.

        Returns:
            List[str]: A list of phonemes and separators, preserving the structure of the input.
        """
        fields = []
        for word in phonemized.split(self.separator.word): # 将整个内容按照word分开
            # Split the word into phonemes and other symbols using regex.
            # Example: "ɐ    m|iː|n?" -> ['ɐ', 'm', '|', 'iː', '|', 'n', '?']
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]  # Exclude phone separators
                + [self.separator.word]  # Add word separator at the end of each word
            )
        
        # Ensure the reconstructed string matches the original phonemized string
        # (excluding phone separators).
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        
        return fields[:-1]  # Return the list without the trailing word separator.

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        signature = None,
        device: Any = None
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model_from_checkpoint(signature, device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels
        self._device = device

        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.codec.encode(wav.to(self.device))
        return [(codes[0], None)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0] # [1,4,T]
        return self.codec.decode(frames)
    


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    if offset != -1 and num_frames!=-1:
        wav, sr = torchaudio.load(audio_path, frame_offset=offset, num_frames=num_frames)
    else:
        wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames
