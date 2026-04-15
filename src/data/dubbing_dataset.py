# cp from https://github.com/jasonppy/VoiceCraft modified by Sungbin Kim
# updated: load lip_feature (*.npy) + video tokens for unified sequence

import os
import omegaconf
import torch
import random
import copy
import logging
import shutil
import json
from typing import Optional
import numpy as np


class dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        cfg: omegaconf.DictConfig,
        name: str = "LRS3",
        manifest_name: str = "manifest",
        drop_long: bool = True,
        audio_min_length: float = 2.0,
        audio_max_length: float = 20.0,
        text_min_length: int = 10,
        text_max_length: int = 400,
        vocab_name: str = "phn2num.txt",
        phn_folder_name: str = "phonemes",
        encodec_folder_name: str = "encodec_16khz_4codebooks",
        dynamic_batching: bool = True,
        pad_x: bool = False,
        sep_special_token: bool = False,
        lip_feature_dir: Optional[str] = None,
        use_ctc_labels: bool = False,
        ctc_phn_folder_name: str = "phonemes_",
        use_alignment_labels: bool = False,
        alignment_label_dir: Optional[str] = None,
        use_split_text_segments: bool = False,
        text_ref_start_token: Optional[int] = None,
        text_target_start_token: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.name = name
        self.manifest_name = manifest_name
        self.drop_long = drop_long
        self.audio_min_length = audio_min_length
        self.audio_max_length = audio_max_length
        self.text_min_length = text_min_length
        self.text_max_length = text_max_length
        self.vocab_name = vocab_name
        self.phn_folder_name = phn_folder_name
        self.encodec_folder_name = encodec_folder_name
        self.dynamic_batching = dynamic_batching
        self.pad_x = pad_x
        self.sep_special_token = sep_special_token
        self.use_ctc_labels = use_ctc_labels
        self.ctc_phn_folder_name = ctc_phn_folder_name
        self.use_alignment_labels = use_alignment_labels
        self.alignment_label_dir = alignment_label_dir
        self.use_split_text_segments = use_split_text_segments
        self.text_ref_start_token = text_ref_start_token
        self.text_target_start_token = text_target_start_token
        self.dataset_dir = cfg.dataset_dir
        self.exp_dir = cfg.exp_dir
        self.encodec_sr = cfg.encodec_sr
        self.n_codebooks = cfg.n_codebooks
        self.special_first = cfg.special_first
        self.n_special = cfg.n_special
        self.text_pad_token = cfg.text_pad_token
        self.audio_pad_token = cfg.audio_pad_token


        assert self.split in ["train", "validation", "test"]
        manifest_fn = os.path.join(self.dataset_dir, self.manifest_name, self.split + ".txt")

        with open(manifest_fn, "r") as rf:
            data = [l.strip().split("\t") for l in rf.readlines()]
        lengths_list = [int(item[-1]) for item in data]

        self.data = []
        self.lengths_list = []
        for d, l in zip(data, lengths_list):
            if l >= self.encodec_sr * self.audio_min_length:
                if self.drop_long and l > self.encodec_sr * self.audio_max_length:
                    continue
                self.data.append(d)
                self.lengths_list.append(l)
        logging.info(f"number of data points for {self.split} split: {len(self.lengths_list)}")

        vocab_fn = os.path.join(self.dataset_dir, self.vocab_name)
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir, exist_ok=True)
        shutil.copy(vocab_fn, os.path.join(self.exp_dir, self.vocab_name))
        with open(vocab_fn, "r") as f:
            temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
            self.phn2num = {item[1]: int(item[0]) for item in temp}

        self.align_phn2num = None
        if self.use_alignment_labels:
            if self.alignment_label_dir is None:
                raise ValueError("alignment_label_dir must be set when use_alignment_labels=True")
            if os.path.isabs(self.alignment_label_dir):
                self.alignment_root = self.alignment_label_dir
            else:
                self.alignment_root = os.path.join(self.dataset_dir, self.alignment_label_dir)
            align_vocab_fn = os.path.join(self.alignment_root, "align_phn2num.txt")
            if not os.path.exists(align_vocab_fn):
                raise FileNotFoundError(align_vocab_fn)
            shutil.copy(align_vocab_fn, os.path.join(self.exp_dir, "align_phn2num.txt"))
            with open(align_vocab_fn, "r") as f:
                temp = [l.strip().split(" ") for l in f.readlines() if len(l) != 0]
                self.align_phn2num = {item[1]: int(item[0]) for item in temp}

        self.symbol_set = set(["<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"])
        self.alignment_silence_labels = {"sil", "sp", "spn", "<eps>", "silence"}
        self.read_split_length()

        if lip_feature_dir:
            if os.path.isabs(lip_feature_dir):
                self.feature_path = lip_feature_dir
            else:
                self.feature_path = os.path.join(self.dataset_dir, lip_feature_dir)
        else:
            self.feature_path = os.path.join(self.dataset_dir, "lip_feature")

    def read_split_length(self):
        self.split_dic = {}
        split_txt = open(os.path.join(self.dataset_dir, "split_len.txt"), "r")
        for line in split_txt:
            txt_name = line.split(",")[0]
            length = int(line.split(",")[1].strip())
            self.split_dic[txt_name] = length

    def __len__(self):
        return len(self.lengths_list)

    def _load_lip_feature(self, feature_path: str) -> torch.Tensor:
        feats = np.load(feature_path)
        if feats.ndim != 2:
            raise ValueError(f"lip_feature must be 2D [T, 1024], got {feats.shape} from {feature_path}")
        if feats.shape[1] != 1024 and feats.shape[0] == 1024:
            feats = feats.T
        if feats.shape[1] != 1024:
            raise ValueError(f"lip_feature dim must be 1024, got {feats.shape} from {feature_path}")
        return torch.from_numpy(feats).float()


    def _candidate_feature_paths(self, item):
        if self.name == "CELEB":
            data_id = item[1].split("__")[0]
            suffix = item[1].split("__")[-1]
            yield os.path.join(self.feature_path, f"{data_id}_{suffix}.npy")
            data_id = "__".join(item[1].split("__")[:2])
            yield os.path.join(self.feature_path, f"{data_id}_{suffix}.npy")
        else:
            base = item[1].split("__")[0] + "_" + item[1].split("__")[-1]
            yield os.path.join(self.feature_path, base + ".npy")

    def _load_phn_enc(self, index):
        item = self.data[index]
        ef = os.path.join(self.dataset_dir, self.encodec_folder_name, item[1] + ".txt")
        if self.use_split_text_segments:
            x = self._load_split_text_phonemes(item)
        else:
            pf = os.path.join(self.dataset_dir, self.phn_folder_name, item[1] + ".txt")
            with open(pf, "r") as p:
                phns = [l.strip() for l in p.readlines()]
            assert len(phns) == 1, phns
            x = [
                self.phn2num[item]
                for item in phns[0].split(" ")
                if item not in self.symbol_set
            ]
        with open(ef, "r") as e:
            encos = [
                l.strip().split()
                for k, l in enumerate(e.readlines())
                if k < self.n_codebooks
            ]

            assert len(encos) == self.n_codebooks, ef
            if self.special_first:
                y = [[int(n) + self.n_special for n in l] for l in encos]
            else:
                y = [[int(n) for n in l] for l in encos]

        split_length = self.split_dic[item[1] + ".txt"]

        return x, y, split_length

    def _load_single_raw_phoneme_ids(self, speaker: str, utt: str) -> list[int]:
        raw_fn = os.path.join(
            self.dataset_dir,
            self.ctc_phn_folder_name,
            f"{speaker}_{utt}.txt",
        )
        with open(raw_fn, "r") as rf:
            lines = [l.strip() for l in rf.readlines()]
        assert len(lines) == 1, lines
        return [
            self.phn2num[token]
            for token in lines[0].split(" ")
            if token not in self.symbol_set
        ]

    def _load_split_text_phonemes(self, item) -> list[int]:
        pair_id = item[1]
        speaker, ref_utt, target_utt = pair_id.split("__")
        if self.text_ref_start_token is None or self.text_target_start_token is None:
            raise ValueError("text_ref_start_token and text_target_start_token must be set when use_split_text_segments=True")

        ref_ids = self._load_single_raw_phoneme_ids(speaker, ref_utt)
        target_ids = self._load_single_raw_phoneme_ids(speaker, target_utt)

        max_content_len = max(0, self.text_max_length - 2)
        if len(target_ids) >= max_content_len:
            kept_target = target_ids[:max_content_len]
            kept_ref = []
        else:
            remaining_ref_budget = max_content_len - len(target_ids)
            kept_ref = ref_ids[-remaining_ref_budget:] if remaining_ref_budget > 0 else []
            kept_target = target_ids

        return [self.text_ref_start_token] + kept_ref + [self.text_target_start_token] + kept_target

    def _load_target_ctc_labels(self, item) -> list[int]:
        pair_id = item[1]
        speaker, _, target_utt = pair_id.split("__")
        ctc_fn = os.path.join(
            self.dataset_dir,
            self.ctc_phn_folder_name,
            f"{speaker}_{target_utt}.txt",
        )
        with open(ctc_fn, "r") as rf:
            lines = [l.strip() for l in rf.readlines()]
        assert len(lines) == 1, lines
        return [
            self.phn2num[token]
            for token in lines[0].split(" ")
            if token not in self.symbol_set
        ]

    def _load_target_alignment_labels(self, item) -> tuple[np.ndarray, int]:
        pair_id = item[1]
        speaker, _, target_utt = pair_id.split("__")
        align_fn = os.path.join(
            self.alignment_root,
            "frame_labels_25hz",
            f"{speaker}_{target_utt}.npz",
        )
        data = np.load(align_fn)
        labels = data["labels"].astype(np.int64)
        frame_count = int(data["frame_count"])
        if labels.shape[0] != frame_count:
            raise ValueError(f"Alignment label length mismatch in {align_fn}: {labels.shape[0]} vs {frame_count}")
        return labels, frame_count

    def _load_target_alignment_occurrence_labels(self, item, frame_count: int) -> tuple[np.ndarray, int]:
        pair_id = item[1]
        speaker, _, target_utt = pair_id.split("__")
        intervals_fn = os.path.join(
            self.alignment_root,
            "phone_intervals",
            f"{speaker}_{target_utt}.json",
        )
        with open(intervals_fn, "r", encoding="utf-8") as rf:
            intervals = json.load(rf)
        labels = np.full((frame_count,), -100, dtype=np.int32)
        occ_idx = 0
        for interval in intervals:
            label = str(interval["label"]).lower()
            if label in self.alignment_silence_labels:
                continue
            begin_idx = max(0, int(np.floor(float(interval["begin"]) * 25.0)))
            end_idx = max(begin_idx + 1, int(np.ceil(float(interval["end"]) * 25.0)))
            end_idx = min(frame_count, end_idx)
            if begin_idx >= frame_count or end_idx <= 0:
                continue
            labels[begin_idx:end_idx] = occ_idx
            occ_idx += 1
        return labels, occ_idx

    def __getitem__(self, index):
        while 1:
            try:
                x, y, split_length = self._load_phn_enc(index)
                break
            except Exception:
                index = random.choice(range(len(self)))
                continue

        x_len, y_len = len(x), len(y[0])

        if x_len == 0 or y_len == 0:
            return {
                "x": None,
                "x_len": None,
                "y": None,
                "y_len": None,
                "y_mask_interval": None,
                "extra_mask_start": None,
            }

        while y_len < self.encodec_sr * self.audio_min_length:
            try:
                index = random.choice(range(len(self)))
                x, y, split_length = self._load_phn_enc(index)
                x_len, y_len = len(x), len(y[0])
            except Exception:
                continue

        if self.drop_long:
            while x_len > self.text_max_length or y_len > self.encodec_sr * self.audio_max_length:
                try:
                    index = random.choice(range(len(self)))
                    x, y, split_length = self._load_phn_enc(index)
                    x_len, y_len = len(x), len(y[0])
                except Exception:
                    continue

        orig_y_len = copy.copy(y_len)
        max_len = int(self.audio_max_length * self.encodec_sr)

        if y_len > max_len:
            audio_start = random.choice(range(0, y_len - max_len))
            for i in range(len(y)):
                y[i] = y[i][audio_start : (audio_start + max_len)]
            y_len = max_len
        else:
            audio_start = 0
            if not self.dynamic_batching:
                pad = [0] * (max_len - y_len) if self.sep_special_token else [self.audio_pad_token] * (max_len - y_len)
                for i in range(len(y)):
                    y[i] = y[i] + pad

        if audio_start > 0 and len(x) > self.text_max_length and not self.use_split_text_segments:
            x = x[int(len(x) * audio_start / orig_y_len) :]
            if len(x) > self.text_max_length:
                x = x[: self.text_max_length]

        x_len = len(x)
        if x_len > self.text_max_length and not self.use_split_text_segments:
            text_start = random.choice(range(0, x_len - self.text_max_length))
            x = x[text_start : text_start + self.text_max_length]
            x_len = self.text_max_length
        elif x_len > self.text_max_length and self.use_split_text_segments:
            x = x[: self.text_max_length]
            x_len = self.text_max_length
        elif self.pad_x and x_len <= self.text_max_length:
            pad = [0] * (self.text_max_length - x_len) if self.sep_special_token else [self.text_pad_token] * (self.text_max_length - x_len)
            x = x + pad

        item = self.data[index]
        feature_loaded = False
        vid = None
        last_error = None
        for feature_path in self._candidate_feature_paths(item):
            try:
                if not os.path.exists(feature_path):
                    raise FileNotFoundError(feature_path)
                vid = self._load_lip_feature(feature_path)
                feature_loaded = True
                break
            except Exception as exc:
                last_error = exc
                continue

        if not feature_loaded:
            logging.error(
                "Error loading lip_feature for index %s, skipping sample. Error: %s",
                index,
                last_error,
            )
            return self.__getitem__(random.choice(range(len(self))))

        vid_lens = vid.shape[0]
        target_len = y_len - split_length
        if abs(target_len - vid_lens * 2) > 1:
            raise ValueError(
                f"Target length mismatch: target_len={target_len}, vid_len={vid_lens}, item={item[1]}"
            )

        ctc_labels = None
        ctc_label_len = None
        if self.use_ctc_labels:
            ctc_labels = self._load_target_ctc_labels(item)
            ctc_label_len = len(ctc_labels)

        align_labels = None
        align_frame_count = None
        if self.use_alignment_labels:
            align_labels, align_frame_count = self._load_target_alignment_labels(item)
            align_occurrence_labels, align_occurrence_count = self._load_target_alignment_occurrence_labels(item, align_frame_count)
        else:
            align_occurrence_labels, align_occurrence_count = None, None

        return {
            "x": torch.LongTensor(x),
            "x_len": x_len,
            "y": torch.LongTensor(y),
            "y_len": y_len,
            "split_length": split_length,
            "v": vid,
            "v_lens": vid_lens,
            "ctc_labels": torch.LongTensor(ctc_labels) if ctc_labels is not None else None,
            "ctc_label_len": ctc_label_len,
            "align_labels": torch.LongTensor(align_labels) if align_labels is not None else None,
            "align_frame_count": align_frame_count,
            "align_occurrence_labels": torch.LongTensor(align_occurrence_labels) if align_occurrence_labels is not None else None,
            "align_occurrence_count": align_occurrence_count,
        }

    def collate(self, batch):
        out = {key: [] for key in batch[0]}
        for item in batch:
            if item["x"] is None:
                continue
            for key, val in item.items():
                out[key].append(val)
        res = {}
        if self.pad_x:
            res["x"] = torch.stack(out["x"], dim=0)
        else:
            res["x"] = torch.nn.utils.rnn.pad_sequence(
                out["x"], batch_first=True, padding_value=self.text_pad_token
            )
        res["x_lens"] = torch.LongTensor(out["x_len"])
        if self.dynamic_batching:
            if out["y"][0].ndim == 2:
                res["y"] = torch.nn.utils.rnn.pad_sequence(
                    [item.transpose(1, 0) for item in out["y"]],
                    padding_value=self.audio_pad_token,
                )
                res["y"] = res["y"].permute(1, 2, 0)
            else:
                assert out["y"][0].ndim == 1, out["y"][0].shape
                res["y"] = torch.nn.utils.rnn.pad_sequence(
                    out["y"], batch_first=True, padding_value=self.audio_pad_token
                )
        else:
            res["y"] = torch.stack(out["y"], dim=0)

        res["v"] = torch.nn.utils.rnn.pad_sequence(out["v"], batch_first=True)
        res["v_lens"] = torch.LongTensor(out["v_lens"])

        res["y_lens"] = torch.LongTensor(out["y_len"])
        res["text_padding_mask"] = (
            torch.arange(res["x"][0].shape[-1]).unsqueeze(0) >= res["x_lens"].unsqueeze(1)
        )

        res["split_lens"] = torch.LongTensor(out["split_length"])

        if out.get("ctc_labels") and out["ctc_labels"][0] is not None:
            res["ctc_labels"] = torch.nn.utils.rnn.pad_sequence(
                out["ctc_labels"], batch_first=True, padding_value=0
            )
            res["ctc_label_lens"] = torch.LongTensor(out["ctc_label_len"])
        if out.get("align_labels") and out["align_labels"][0] is not None:
            res["align_labels"] = torch.nn.utils.rnn.pad_sequence(
                out["align_labels"], batch_first=True, padding_value=-100
            )
            res["align_frame_counts"] = torch.LongTensor(out["align_frame_count"])
        if out.get("align_occurrence_labels") and out["align_occurrence_labels"][0] is not None:
            res["align_occurrence_labels"] = torch.nn.utils.rnn.pad_sequence(
                out["align_occurrence_labels"], batch_first=True, padding_value=-100
            )
            res["align_occurrence_counts"] = torch.LongTensor(out["align_occurrence_count"])
        return res
