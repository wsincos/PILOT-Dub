import argparse
import glob
import logging
import os
import re
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
import tqdm

from src.data.tokenizer import AudioTokenizer, TextTokenizer, tokenize_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHN2NUM = PROJECT_ROOT / "artifacts" / "pretrained_models" / "tokenizers" / "phn2num.txt"

PUNC2SYM = {
    " <COMMA>": ",",
    " <PERIOD>": ".",
    " <QUESTIONMARK>": "?",
    " <EXCLAMATIONPOINT>": "!",
}
GAR2SYM = {
    "<SIL>": "#%#",
    "<MUSIC>": "##%",
    "<NOISE>": "%%#",
    "<OTHER>": "%#%",
}
WORD2SYM = {
    "h æ ʃ h ɐ ʃ p ɚ s ɛ n t": "<MUSIC>",
    "h æ ʃ p ɚ s ɛ n t h æ ʃ": "<SIL>",
    "p ɚ s ɛ n t h ɐ ʃ p ɚ s ɛ n t": "<OTHER>",
    "p ɚ s ɛ n t p ɚ s ɛ n t h æ ʃ": "<NOISE>",
}
FORBIDDEN_WORDS = {"#%#", "##%", "%%#", "%#%"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phonemize LRS3 transcripts and extract single-utterance EnCodec tokens."
    )
    parser.add_argument("--save_dir", type=str, default="../samples/trainval_preprocess")
    parser.add_argument("--root_dir", type=str, default="../samples/trainval")
    parser.add_argument("--split_name", type=str, default="trainval")
    parser.add_argument("--encodec_model_path", type=str, default="../pretrained_models/encodec.th")
    parser.add_argument(
        "--encodec_device",
        type=str,
        default="cuda",
        help="Device for EnCodec tokenization. Use cpu for smoke tests when GPU memory is tight.",
    )
    parser.add_argument("--phn2num_path", type=str, default=str(DEFAULT_PHN2NUM))
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument(
        "--mega_batch_size",
        type=int,
        default=100,
        help="Number of samples in each mega batch for multiprocess dataloading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for EnCodec encoding across the active device.",
    )
    parser.add_argument("--model_sr", type=int, default=16000, help="Encodec input audio sample rate")
    parser.add_argument("--downsample_rate", type=int, default=320, help="Encodec downsample rate")
    parser.add_argument("--model_code_sr", type=int, default=50, help="Encodec model code sample rate")
    parser.add_argument(
        "--len_cap",
        type=float,
        default=35.0,
        help="Drop audios that are longer than this duration in seconds",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=30000,
        help="Max audio length in samples before splitting a batch to avoid OOM",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing phoneme / EnCodec files when they already exist.",
    )
    return parser.parse_args()


def load_phn2num(phn2num_path: str) -> dict[str, int]:
    with open(phn2num_path, "r", encoding="utf-8") as f:
        return {
            phn: int(idx)
            for idx, phn in (line.strip().split(" ", 1) for line in f if line.strip())
        }


def copy_phn2num(phn2num_path: str, save_dir: str) -> str:
    dst = os.path.join(save_dir, "phn2num.txt")
    shutil.copy2(phn2num_path, dst)
    return dst


def sort_by_audio_len(lens: np.ndarray, model_code_sr: int):
    inds = np.argsort(lens).tolist()
    logging.info("longest: %s encodec codes, %.2f sec.", lens[inds[-1]] * model_code_sr, lens[inds[-1]])
    logging.info("shortest: %s encodec codes, %.2f sec.", lens[inds[0]] * model_code_sr, lens[inds[0]])
    logging.info(
        "median: %s encodec codes, %.2f sec.",
        lens[inds[len(inds) // 2]] * model_code_sr,
        lens[inds[len(inds) // 2]],
    )
    logging.info(
        "95 percentile longest: %s encodec codes, %.2f sec.",
        lens[inds[int(len(inds) * 0.95)]] * model_code_sr,
        lens[inds[int(len(inds) * 0.95)]],
    )
    return inds[::-1]


def write_array_to_txt_file(array, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for a in array[:-1]:
            f.write(" ".join(map(str, a)) + "\n")
        f.write(" ".join(map(str, array[-1])))


def read_lrs3_text(text_path: str) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if "Text:" not in first_line:
        raise ValueError(f"Unexpected transcript format in {text_path}")
    return first_line.split("Text:", 1)[1].strip()


def normalize_text(raw_text: str) -> str:
    return re.sub(r"[^a-zA-Z]+$", "", raw_text) + " <PERIOD>"


def phonemize_to_sequence(raw_text: str, text_tokenizer: TextTokenizer) -> str:
    text = normalize_text(raw_text)
    if any(word in FORBIDDEN_WORDS for word in text.split(" ")):
        raise ValueError(f"Transcript contains forbidden placeholder token: {text}")
    replacement_map = dict(PUNC2SYM)
    replacement_map.update(GAR2SYM)
    for src, dst in replacement_map.items():
        text = text.replace(src, dst)
    phn = tokenize_text(text_tokenizer, text)
    phn_seq = " ".join(phn)
    for src, dst in WORD2SYM.items():
        phn_seq = phn_seq.replace(src, dst)
    return phn_seq


def validate_tokens(phn_seq: str, phn2num: dict[str, int], text_path: str):
    missing = sorted({token for token in phn_seq.split(" ") if token and token not in phn2num})
    if missing:
        raise ValueError(f"Found OOV phoneme tokens in {text_path}: {missing}")


def list_lrs3_text_files(root_dir: str, split_name: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root_dir, split_name, "*", "*.txt")))


def build_single_utterance_phonemes(
    text_paths: list[str],
    phn_save_root: str,
    text_tokenizer: TextTokenizer,
    phn2num: dict[str, int],
    skip_existing: bool,
) -> list[tuple[str, str]]:
    valid_items: list[tuple[str, str]] = []
    all_lens: list[int] = []
    for text_path in tqdm.tqdm(text_paths):
        speaker = os.path.basename(os.path.dirname(text_path))
        utt = os.path.splitext(os.path.basename(text_path))[0]
        segment_id = f"{speaker}_{utt}"
        save_fn = os.path.join(phn_save_root, segment_id + ".txt")

        if skip_existing and os.path.isfile(save_fn):
            with open(save_fn, "r", encoding="utf-8") as f:
                phn_seq = f.readline().strip()
        else:
            raw_text = read_lrs3_text(text_path)
            try:
                phn_seq = phonemize_to_sequence(raw_text, text_tokenizer)
            except ValueError as exc:
                logging.warning("Skipping %s: %s", text_path, exc)
                continue
            validate_tokens(phn_seq, phn2num, text_path)
            with open(save_fn, "w", encoding="utf-8") as f:
                f.write(phn_seq)

        validate_tokens(phn_seq, phn2num, text_path)
        all_lens.append(len([token for token in phn_seq.split(" ") if token]))
        valid_items.append((segment_id, text_path))

    if not all_lens:
        raise RuntimeError("No valid transcripts were produced.")

    logging.info("valid utterances after phonemization: %d / %d", len(valid_items), len(text_paths))
    logging.info("phoneme sequence stats:")
    logging.info("longest: %d", max(all_lens))
    logging.info("shortest: %d", min(all_lens))
    logging.info("median: %.2f", np.quantile(all_lens, 0.5))
    logging.info("95 percentile longest: %.2f", np.quantile(all_lens, 0.95))
    return valid_items


class SingleUtteranceDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[str, str]]):
        super().__init__()
        self.items = items

    def __len__(self):
        return len(self.items)

    def load_audio(self, text_path: str):
        audio_file, _ = librosa.load(text_path.replace(".txt", ".wav"), sr=16000)
        duration = librosa.get_duration(y=audio_file, sr=16000)
        return torch.from_numpy(audio_file).float(), duration

    def __getitem__(self, ind):
        segment_id, text_path = self.items[ind]
        audio, duration = self.load_audio(text_path)
        return segment_id, audio, 16000, duration

    def collate(self, batch):
        res = {"segment_id": [], "audio": [], "sr": [], "duration": []}
        for item in batch:
            if item[0] is None:
                continue
            res["segment_id"].append(item[0])
            res["audio"].append(item[1])
            res["sr"].append(item[2])
            res["duration"].append(item[3])
        return res


def extract_encodec_tokens(args, items: list[tuple[str, str]], codes_save_root: str):
    model = AudioTokenizer(args.encodec_model_path, device=args.encodec_device)
    dataset = SingleUtteranceDataset(items)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.mega_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.n_workers,
        collate_fn=dataset.collate,
    )

    mega_n_steps = int(np.ceil(len(items) / args.mega_batch_size))
    logging.info("encodec encoding %d valid utterances in %d mega steps", len(items), mega_n_steps)

    for m, mega_batch in enumerate(loader):
        logging.info("====================================")
        logging.info("now processing mega step %d/%d", m + 1, mega_n_steps)
        lengths = np.array(mega_batch["duration"])
        sorted_inds = sort_by_audio_len(lengths, args.model_code_sr)

        for j in range(len(sorted_inds))[::-1]:
            if lengths[sorted_inds[j]] < 0.2 or lengths[sorted_inds[j]] > args.len_cap:
                del sorted_inds[j]

        n_steps = int(np.ceil(len(sorted_inds) / args.batch_size))
        for n in tqdm.tqdm(range(n_steps), disable=True):
            inds_used = sorted_inds[n * args.batch_size : (n + 1) * args.batch_size]
            audio_batch = [mega_batch["audio"][idx] for idx in inds_used]
            segment_id_batch = [mega_batch["segment_id"][idx] for idx in inds_used]
            padded_wav = torch.nn.utils.rnn.pad_sequence(audio_batch, batch_first=True).unsqueeze(1)
            all_lens = [lengths[idx] for idx in inds_used]

            with torch.no_grad():
                input_wav = padded_wav.to(model.device)
                if max(all_lens) > args.max_len and len(all_lens) > 1:
                    codes = []
                    codes.append(model.encode(input_wav[: len(input_wav) // 2])[0][0].cpu())
                    codes.append(model.encode(input_wav[len(input_wav) // 2 :])[0][0].cpu())
                    codes = torch.cat(codes, dim=0)
                else:
                    encoded_frames = model.encode(input_wav)
                    codes = encoded_frames[0][0].cpu()

            for i, length in enumerate(all_lens):
                save_fn = os.path.join(codes_save_root, segment_id_batch[i] + ".txt")
                if args.skip_existing and os.path.isfile(save_fn):
                    continue
                actual_len = round(length * args.model_code_sr)
                cur_code = codes[i, :, :actual_len].tolist()
                write_array_to_txt_file(cur_code, save_fn)


def main():
    formatter = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    args = parse_args()

    phn_save_root = os.path.join(args.save_dir, "phonemes_")
    codes_save_root = os.path.join(args.save_dir, "encodec_16khz_4codebooks_")
    os.makedirs(phn_save_root, exist_ok=True)
    os.makedirs(codes_save_root, exist_ok=True)

    phn2num = load_phn2num(args.phn2num_path)
    copied_phn2num = copy_phn2num(args.phn2num_path, args.save_dir)
    logging.info("using phn2num from %s", args.phn2num_path)
    logging.info("copied phn2num to %s", copied_phn2num)

    text_paths = list_lrs3_text_files(args.root_dir, args.split_name)
    if not text_paths:
        raise RuntimeError(f"No LRS3 transcripts found under {args.root_dir}/{args.split_name}")

    text_tokenizer = TextTokenizer()
    valid_items = build_single_utterance_phonemes(
        text_paths=text_paths,
        phn_save_root=phn_save_root,
        text_tokenizer=text_tokenizer,
        phn2num=phn2num,
        skip_existing=args.skip_existing,
    )
    extract_encodec_tokens(args, valid_items, codes_save_root)


if __name__ == "__main__":
    main()
