from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch


VIS_SIL = 0
VIS_BILABIAL = 1
VIS_LABIODENTAL = 2
VIS_DENTAL_ALVEOLAR_FRIC = 3
VIS_POSTALVEOLAR_AFFRICATE = 4
VIS_ALVEOLAR = 5
VIS_VELAR_GLOTTAL = 6
VIS_APPROX = 7
VIS_ROUNDED_VOWEL = 8
VIS_FRONT_VOWEL = 9
VIS_OPEN_CENTRAL_VOWEL = 10
NUM_VISEME_CLASSES = 11


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip()


def _normalize_arpabet(symbol: str) -> str:
    s = symbol.strip().upper()
    while s and s[-1].isdigit():
        s = s[:-1]
    return s


def map_ipa_symbol_to_viseme(symbol: str) -> int:
    s = _normalize_symbol(symbol)
    if s in {
        "<SIL>",
        "<MUSIC>",
        "<NOISE>",
        "<OTHER>",
        "_",
        ",",
        ".",
        "?",
        ";",
        ":",
        "!",
        "\"",
        "«",
        "»",
        "¿",
        "¡",
        "…",
        "—",
        "1",
    }:
        return VIS_SIL

    if s in {"p", "b", "m"}:
        return VIS_BILABIAL
    if s in {"f", "v"}:
        return VIS_LABIODENTAL
    if s in {"θ", "ð", "s", "z"}:
        return VIS_DENTAL_ALVEOLAR_FRIC
    if s in {"ʃ", "ʒ", "tʃ", "dʒ", "tɕ"}:
        return VIS_POSTALVEOLAR_AFFRICATE
    if s in {"t", "d", "n", "nʲ", "l", "ɾ"}:
        return VIS_ALVEOLAR
    if s in {"k", "ɡ", "g", "ŋ", "h", "x", "q", "kh", "ç", "ɬ", "ʔ"}:
        return VIS_VELAR_GLOTTAL
    if s in {"w", "j", "ɹ", "r"}:
        return VIS_APPROX
    if s in {
        "u",
        "uː",
        "ʊ",
        "o",
        "oː",
        "oʊ",
        "əʊ",
        "ɔ",
        "ɔː",
        "ɔːɹ",
        "ɔɪ",
        "oːɹ",
        "ʊɹ",
        "aʊ",
        "aw",
    }:
        return VIS_ROUNDED_VOWEL
    if s in {
        "i",
        "iː",
        "iːː",
        "ɪ",
        "ɪː",
        "e",
        "eɪ",
        "ɛ",
        "ɛː",
        "ɛɹ",
        "æ",
        "ææ",
        "aɪ",
        "aɪɚ",
        "aɪə",
        "iə",
        "ɪɹ",
    }:
        return VIS_FRONT_VOWEL
    if s in {
        "ɑ",
        "ɑː",
        "ɑːɹ",
        "ɐ",
        "ɐɐ",
        "ʌ",
        "ə",
        "əl",
        "ɚ",
        "ɜː",
        "ᵻ",
        "əʊ",
        "ɔɹ",
        "əɹ",
    }:
        return VIS_OPEN_CENTRAL_VOWEL
    return VIS_SIL


def map_arpabet_symbol_to_viseme(symbol: str) -> int:
    s = _normalize_arpabet(symbol)
    if s in {"SIL", "SP", "SPN", "<EPS>", "SILENCE"}:
        return VIS_SIL
    if s in {"P", "B", "M"}:
        return VIS_BILABIAL
    if s in {"F", "V"}:
        return VIS_LABIODENTAL
    if s in {"TH", "DH", "S", "Z"}:
        return VIS_DENTAL_ALVEOLAR_FRIC
    if s in {"SH", "ZH", "CH", "JH"}:
        return VIS_POSTALVEOLAR_AFFRICATE
    if s in {"T", "D", "N", "L"}:
        return VIS_ALVEOLAR
    if s in {"K", "G", "NG", "HH"}:
        return VIS_VELAR_GLOTTAL
    if s in {"R", "Y", "W"}:
        return VIS_APPROX
    if s in {"UW", "UH", "OW", "OY", "AO", "AW"}:
        return VIS_ROUNDED_VOWEL
    if s in {"IY", "IH", "EY", "EH", "AE", "AY"}:
        return VIS_FRONT_VOWEL
    if s in {"AA", "AH", "ER"}:
        return VIS_OPEN_CENTRAL_VOWEL
    return VIS_SIL


def build_text_token_to_viseme_table(vocab_path: str, table_size: int) -> torch.Tensor:
    table = torch.full((table_size,), VIS_SIL, dtype=torch.long)
    with open(Path(vocab_path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_str, symbol = line.split(" ", 1)
            idx = int(idx_str)
            if 0 <= idx < table_size:
                table[idx] = map_ipa_symbol_to_viseme(symbol)
    return table


def build_align_label_to_viseme_table(vocab_path: str) -> torch.Tensor:
    mapping: Dict[int, int] = {}
    max_idx = -1
    with open(Path(vocab_path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            idx_str, symbol = line.split(" ", 1)
            idx = int(idx_str)
            mapping[idx] = map_arpabet_symbol_to_viseme(symbol)
            max_idx = max(max_idx, idx)
    table = torch.full((max_idx + 1,), VIS_SIL, dtype=torch.long)
    for idx, vis in mapping.items():
        table[idx] = vis
    return table
