from omegaconf import OmegaConf, open_dict
import hydra

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

import os
import sys
from pathlib import Path
import random
import numpy as np
import glob
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio

from src.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from src.utils.utils import dict_from_config


def patch_multiprocessing_lock():
    import multiprocessing
    import threading

    try:
        multiprocessing.Lock()
    except PermissionError:
        multiprocessing.Lock = threading.Lock


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(cfg):
    device = cfg.device
    if isinstance(device, (int, float)):
        device = str(int(device))
    if isinstance(device, str) and device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        with open_dict(cfg):
            cfg.device = "cuda"
        device = "cuda"
    elif isinstance(device, str) and device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        with open_dict(cfg):
            cfg.device = "cuda"
        device = "cuda"

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        with open_dict(cfg):
            cfg.device = "cpu"
        return "cpu"
    return device


def load_lip_feature(feature_path: str) -> torch.Tensor:
    feats = np.load(feature_path)
    if feats.ndim != 2:
        raise ValueError(f"lip_feature must be 2D [T, 1024], got {feats.shape} from {feature_path}")
    if feats.shape[1] != 1024 and feats.shape[0] == 1024:
        feats = feats.T
    if feats.shape[1] != 1024:
        raise ValueError(f"lip_feature dim must be 1024, got {feats.shape} from {feature_path}")
    return torch.from_numpy(feats).float()


def is_valid_lip_feature_file(feature_path: str) -> bool:
    try:
        feats = np.load(feature_path, mmap_mode="r")
    except Exception:
        return False
    if feats.ndim != 2:
        return False
    return feats.shape[1] == 1024 or feats.shape[0] == 1024


def resolve_avhubert_ckpt(cfg):
    ckpt_path = OmegaConf.select(cfg, "artifacts.avhubert_path", default=None)
    if ckpt_path is None:
        ckpt_path = OmegaConf.select(cfg, "avhubert_path", default=None)
    if ckpt_path is None:
        raise ValueError("avhubert checkpoint not found in cfg.artifacts.avhubert_path or cfg.avhubert_path")
    return ckpt_path


def load_tokenizers(cfg, device):
    text_cfg = dict_from_config(cfg.tokenizer.text)
    audio_cfg = dict_from_config(cfg.tokenizer.audio)
    if device == "cpu":
        audio_cfg["device"] = "cpu"
    audio_tokenizer = AudioTokenizer(**audio_cfg)
    text_tokenizer = TextTokenizer(**text_cfg)
    return audio_tokenizer, text_tokenizer


def load_phn2num(cfg):
    vocab_path = cfg.get("vocab_path", None)
    if vocab_path is None:
        vocab_path = os.path.join(cfg.pretrained_models_dir, "tokenizers", "phn2num.txt")
    phn2num = {}
    with open(vocab_path, "r") as f:
        for line in f:
            idx, phn = line.strip().split()
            phn2num[phn] = int(idx)
    return phn2num


def read_lrs3_text(text_path: str) -> str:
    line = open(text_path, "r").readline().strip()
    if "Text: " in line:
        line = line.split("Text: ", 1)[1]
    return line.strip().lower()


def _normalize_text_segment(text: str) -> str:
    text = text.strip().lower()
    if text and text[-1] not in ".!?;:":
        text = f"{text}."
    return text


def _cfg_select(cfg, path, default=None):
    try:
        return OmegaConf.select(cfg, path, default=default)
    except Exception:
        return default


def use_split_text_segments(cfg) -> bool:
    return bool(_cfg_select(cfg, "dataset.use_split_text_segments", default=False))


def build_text_tokens_for_model(text_tokenizer, phn2num, ref_text, target_text, cfg):
    ref_text = _normalize_text_segment(ref_text)
    target_text = _normalize_text_segment(target_text)

    if use_split_text_segments(cfg):
        ref_start = _cfg_select(cfg, "text_ref_start_token", default=None)
        if ref_start is None:
            ref_start = _cfg_select(cfg, "dataset.text_ref_start_token", default=None)
        target_start = _cfg_select(cfg, "text_target_start_token", default=None)
        if target_start is None:
            target_start = _cfg_select(cfg, "dataset.text_target_start_token", default=None)
        if ref_start is None or target_start is None:
            raise ValueError(
                "Split text inference is enabled, but text_ref_start_token/text_target_start_token "
                "are missing from the config."
            )

        ref_ids = [phn2num[phn] for phn in tokenize_text(text_tokenizer, text=ref_text) if phn in phn2num]
        target_ids = [
            phn2num[phn] for phn in tokenize_text(text_tokenizer, text=target_text) if phn in phn2num
        ]
        text_ids = [int(ref_start)] + ref_ids + [int(target_start)] + target_ids
    else:
        combined_text = f"{ref_text} {target_text}".strip()
        text_ids = [
            phn2num[phn]
            for phn in tokenize_text(text_tokenizer, text=combined_text)
            if phn in phn2num
        ]

    if not text_ids:
        raise ValueError("No valid phoneme ids were built from the provided ref/target texts.")
    return torch.LongTensor(text_ids).unsqueeze(0)


def normalize_state_dict_from_origin(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k

        if "adapting_layer.0.0." in k:
            new_k = k.replace("adapting_layer.0.0.", "adapting_layer.0.")
        elif "adapting_layer.0.2." in k:
            new_k = k.replace("adapting_layer.0.2.", "adapting_layer.2.")
        elif k.startswith("audio_embedding."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                new_k = k.replace(
                    f"audio_embedding.{parts[1]}",
                    f"audio_embedding.audio_embeddings.{parts[1]}",
                )
        elif k.startswith("predict_layer."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                new_k = k.replace(f"predict_layer.{parts[1]}", f"predict_layer.heads.{parts[1]}")
        elif k == "mask_embedding":
            new_k = "mask_embedding.embedding"
        elif k == "eog":
            new_k = "processor.eog"
        elif k == "eos":
            new_k = "processor.eos"

        new_state_dict[new_k] = v
    return new_state_dict


def _resolve_ckpt_path(cfg):
    ckpt_path = OmegaConf.select(cfg, "ckpt_path", default=None)
    if ckpt_path:
        return ckpt_path
    for key in ("load_original_model_from", "original_model"):
        fallback = OmegaConf.select(cfg, key, default=None)
        if fallback:
            return fallback
    return None


def _strip_state_dict_prefixes(state_dict):
    prefixes = ("model.module.", "model._orig_mod.", "model.", "module.", "_orig_mod.")
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }
    return state_dict


def _looks_like_origin_state_dict(state_dict):
    for k in state_dict:
        if k.startswith(("adapting_layer.0.0.", "adapting_layer.0.2.")):
            return True
        if k.startswith("audio_embedding."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                return True
        if k.startswith("predict_layer."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                return True
        if k in ("mask_embedding", "eog", "eos"):
            return True
    return False


def load_model(cfg, device):
    model = hydra.utils.instantiate(cfg.model)
    ckpt_path = _resolve_ckpt_path(cfg)
    if not ckpt_path:
        raise ValueError("ckpt_path is required for inference.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    phn2num = None
    origin_hint = False
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            origin_hint = True
            state_dict = checkpoint["model"]
            phn2num = checkpoint.get("phn2num")
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            phn2num = checkpoint.get("phn2num")
    else:
        state_dict = checkpoint

    state_dict = _strip_state_dict_prefixes(state_dict)
    if origin_hint or _looks_like_origin_state_dict(state_dict):
        state_dict = normalize_state_dict_from_origin(state_dict)

    # DEBUG: print checkpoint/model mismatching details before loading
    if OmegaConf.select(cfg, "print_missmatching", default=False):
        model_state_dict = model.state_dict()
        missing_keys = [k for k in model_state_dict.keys() if k not in state_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in model_state_dict]
        shape_mismatches = []
        dtype_mismatches = []
        for k, model_tensor in model_state_dict.items():
            ckpt_tensor = state_dict.get(k)
            if ckpt_tensor is None:
                continue
            if ckpt_tensor.shape != model_tensor.shape:
                shape_mismatches.append((k, tuple(ckpt_tensor.shape), tuple(model_tensor.shape)))
                continue
            if ckpt_tensor.dtype != model_tensor.dtype:
                dtype_mismatches.append((k, ckpt_tensor.dtype, model_tensor.dtype))

        if missing_keys or unexpected_keys or shape_mismatches or dtype_mismatches:
            print(
                "Checkpoint/model mismatch summary: missing=%d, unexpected=%d, "
                "shape_mismatch=%d, dtype_mismatch=%d"
                % (
                    len(missing_keys),
                    len(unexpected_keys),
                    len(shape_mismatches),
                    len(dtype_mismatches),
                )
            )
        if missing_keys:
            print("Missing keys (in model but not in checkpoint):")
            for k in missing_keys:
                print(f"  - {k}")
        if unexpected_keys:
            print("Unexpected keys (in checkpoint but not in model):")
            for k in unexpected_keys:
                print(f"  - {k}")
        if shape_mismatches:
            print("Shape-mismatched keys (checkpoint vs model):")
            for k, ckpt_shape, model_shape in shape_mismatches:
                print(f"  - {k}: ckpt={ckpt_shape}, model={model_shape}")
        if dtype_mismatches:
            print("Dtype-mismatched keys (checkpoint vs model):")
            for k, ckpt_dtype, model_dtype in dtype_mismatches:
                print(f"  - {k}: ckpt={ckpt_dtype}, model={model_dtype}")

    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint: {ckpt_path}. "
            "This usually means the model config (e.g., fast vs default) "
            "does not match the checkpoint architecture."
        ) from exc

    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "Warning: checkpoint keys mismatch. Missing=%d, unexpected=%d"
            % (len(incompatible.missing_keys), len(incompatible.unexpected_keys))
        )

    model.to(device)
    model.eval()
    if phn2num is None:
        phn2num = load_phn2num(cfg)
    return model, phn2num


def resolve_full_video_path(cfg):
    full_video = OmegaConf.select(cfg, "tar_video_origin", default=None)
    if full_video is None:
        full_video = OmegaConf.select(cfg, "tar_video_full", default=None)
    if full_video is None:
        full_video = cfg.tar_video
    return full_video


def resolve_lip_video_path(cfg):
    lip_video = OmegaConf.select(cfg, "tar_lip_video", default=None)
    if lip_video is None:
        lip_video = OmegaConf.select(cfg, "tar_video_lip", default=None)
    if lip_video is None:
        lip_video = cfg.tar_video
    return lip_video


def lip_feature_name_from_wav(wav_path: str) -> str:
    base = os.path.splitext(os.path.basename(wav_path))[0]
    return f"{base}_lip_feature.npy"


def lip_feature_name_from_video(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base}.npy"


def resolve_lip_feature_dir(cfg):
    lip_feature_dir = OmegaConf.select(cfg, "lip_feature_dir", default=None)
    if lip_feature_dir is None:
        lip_feature_dir = OmegaConf.select(cfg, "target_lip_feature_dir", default=None)
    return lip_feature_dir


def resolve_output_lip_feature_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "_lip_features")


def resolve_lip_feature_path(cfg, wav_path: str, target_video_path: str, output_dir: str, device):
    for key in ("tar_lip_feature", "tar_video_npy", "tar_lip_npy", "tar_video_feature"):
        candidate = OmegaConf.select(cfg, key, default=None)
        if candidate and os.path.exists(candidate):
            if is_valid_lip_feature_file(candidate):
                return candidate
            raise ValueError(
                f"Configured lip feature path is not a valid AV-HuBERT feature [T, 1024]: {candidate}"
            )

    output_lip_feature_dir = resolve_output_lip_feature_dir(output_dir)
    output_path = None
    legacy_output_path = None
    if target_video_path:
        output_path = os.path.join(output_lip_feature_dir, lip_feature_name_from_video(target_video_path))
    if wav_path:
        legacy_output_path = os.path.join(output_lip_feature_dir, lip_feature_name_from_wav(wav_path))
    for candidate in (output_path, legacy_output_path):
        if candidate and os.path.exists(candidate) and is_valid_lip_feature_file(candidate):
            return candidate

    # Prefer features keyed by the target video stem. This is the correct cache key
    # for evaluation because different target videos can share the same reference audio.
    if target_video_path:
        target_dir = os.path.dirname(target_video_path)
        target_base = os.path.splitext(os.path.basename(target_video_path))[0]
        for candidate in (
            os.path.join(target_dir, f"{target_base}.npy"),
            os.path.join(target_dir, f"{target_base}_lip_feature.npy"),
        ):
            if os.path.exists(candidate) and is_valid_lip_feature_file(candidate):
                return candidate

    wav_feature = os.path.join(os.path.dirname(wav_path), lip_feature_name_from_wav(wav_path))
    if os.path.exists(wav_feature) and is_valid_lip_feature_file(wav_feature):
        return wav_feature

    lip_feature_dir = resolve_lip_feature_dir(cfg)
    if lip_feature_dir:
        candidate_names = []
        if target_video_path:
            target_base = os.path.splitext(os.path.basename(target_video_path))[0]
            candidate_names.extend((f"{target_base}.npy", f"{target_base}_lip_feature.npy"))
        wav_base = os.path.splitext(os.path.basename(wav_path))[0]
        candidate_names.extend((f"{wav_base}.npy", lip_feature_name_from_wav(wav_path)))
        for candidate_name in candidate_names:
            candidate = os.path.join(lip_feature_dir, candidate_name)
            if os.path.exists(candidate) and is_valid_lip_feature_file(candidate):
                return candidate

    if (
        target_video_path.endswith(".npy")
        and os.path.exists(target_video_path)
        and is_valid_lip_feature_file(target_video_path)
    ):
        return target_video_path

    if target_video_path.endswith(".mp4") and os.path.exists(target_video_path):
        os.makedirs(output_lip_feature_dir, exist_ok=True)
        if os.path.exists(output_path) and is_valid_lip_feature_file(output_path):
            return output_path
        if os.path.exists(legacy_output_path) and is_valid_lip_feature_file(legacy_output_path):
            return legacy_output_path
        device_str = device if isinstance(device, str) else getattr(device, "type", "cpu")
        ckpt_path = resolve_avhubert_ckpt(cfg)
        base_cmd = [
            "python",
            "src/data/utils/extract_avhubert_features.py",
            "--ckpt_path",
            ckpt_path,
            "--input",
            target_video_path,
            "--output_dir",
            output_lip_feature_dir,
            "--ext",
            ".mp4",
            "--skip_existing",
        ]
        attempted_devices = [device_str]
        if device_str.startswith("cuda"):
            attempted_devices.append("cpu")
        last_error = None
        last_result = None
        for extract_device in attempted_devices:
            cmd = [*base_cmd, "--device", extract_device]
            result = subprocess.run(cmd, capture_output=True, text=True)
            last_result = result
            if result.returncode == 0:
                break
            last_error = subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
            combined_output = f"{result.stdout}\n{result.stderr}".lower()
            if extract_device.startswith("cuda") and "out of memory" in combined_output:
                print(
                    f"AV-HuBERT feature extraction hit CUDA OOM for {target_video_path}; retrying on CPU.",
                    flush=True,
                )
                continue
            raise RuntimeError(
                f"extract_avhubert_features failed for {target_video_path} on device={extract_device}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ) from last_error
        else:
            raise RuntimeError(
                f"extract_avhubert_features failed for {target_video_path} after retries.\n"
                f"stdout:\n{getattr(last_error, 'output', '')}\n"
                f"stderr:\n{getattr(last_error, 'stderr', '')}"
            ) from last_error
        generated_path = os.path.join(output_lip_feature_dir, lip_feature_name_from_video(target_video_path))
        if not os.path.exists(generated_path):
            raise RuntimeError(
                f"extract_avhubert_features completed without creating {generated_path}.\n"
                f"stdout:\n{getattr(last_result, 'stdout', '')}\n"
                f"stderr:\n{getattr(last_result, 'stderr', '')}"
            )
        if not is_valid_lip_feature_file(generated_path):
            raise RuntimeError(
                f"extract_avhubert_features created an invalid lip feature at {generated_path}.\n"
                f"stdout:\n{getattr(last_result, 'stdout', '')}\n"
                f"stderr:\n{getattr(last_result, 'stderr', '')}"
            )
        return generated_path

    raise FileNotFoundError(
        "lip_feature .npy not found. Provide tar_lip_feature or lip_feature_dir, "
        "or pass an mp4 to generate the feature."
    )


def process_inputs(audio_tokenizer, text_tokenizer, phn2num, cfg, target_video_path, output_dir, device):
    encoded_frames = tokenize_audio(audio_tokenizer, cfg.src_audio, offset=0)
    audio_tokens = encoded_frames[0][0]

    src_text = read_lrs3_text(cfg.src_text)
    tar_text = read_lrs3_text(cfg.tar_text)
    text_tokens = build_text_tokens_for_model(text_tokenizer, phn2num, src_text, tar_text, cfg)
    lip_feature_path = resolve_lip_feature_path(cfg, cfg.src_audio, target_video_path, output_dir, device)
    target_video = load_lip_feature(lip_feature_path)
    return audio_tokens, text_tokens, target_video


@torch.no_grad()
def inference_dubbing_sample(model, original_audio, text_tokens, target_video, device, decode_config):
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    if decode_config["sample_batch_size"] >= 1:
        concat_frames, gen_frames = model.generate(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio.to(device),
            target_video.to(device),
            top_k=decode_config["top_k"],
            top_p=decode_config["top_p"],
            temperature=decode_config["temperature"],
            stop_repetition=decode_config["stop_repetition"],
            kvcache=decode_config["kvcache"],
            silence_tokens=decode_config["silence_tokens"],
        )

    return concat_frames, gen_frames


def replace_audio_in_video(video_path, audio_path, output_path):
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        "-map",
        "0:v",
        "-map",
        "1:a",
        output_path,
    ]
    subprocess.run(command, check=True)


def audio_decode(audio_tokenizer, root_dir, target_video, device, output_suffix=""):
    gen_list = glob.glob(os.path.join(root_dir, "gen*.npy"))
    for gen_path in gen_list:
        gen = np.load(gen_path)
        gen = torch.from_numpy(gen).to(device)
        if len(gen.shape) == 2:
            gen = gen.unsqueeze(0)
        gen_audio = audio_tokenizer.decode([(gen, None)])
        gen_audio = gen_audio[0].cpu()

        seg_save_fn_gen = os.path.join(root_dir, gen_path.split("/")[-1].replace(".npy", ".wav"))
        torchaudio.save(seg_save_fn_gen, gen_audio.detach(), 16000)
        output_file = os.path.join(
            root_dir,
            gen_path.split("/")[-1].replace(".npy", f"{output_suffix}.mp4"),
        )
        replace_audio_in_video(target_video, seg_save_fn_gen, output_file)


def save_results(
    result_dir,
    input_video,
    gen_audio,
    audio_tokenizer,
    full_video_path,
    lip_video_path,
    device,
):
    save_dir = os.path.join(f"{result_dir}", input_video.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)

    for g_idx, gg in enumerate(gen_audio):
        np.save(os.path.join(save_dir, f"gen_{g_idx}.npy"), gg.cpu().numpy())
    audio_decode(audio_tokenizer, save_dir, full_video_path, device, output_suffix="")
    audio_decode(audio_tokenizer, save_dir, lip_video_path, device, output_suffix="_lip")


@hydra.main(config_path="../configs", config_name="inference/inference", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)
    patch_multiprocessing_lock()

    device = resolve_device(cfg)
    audio_tokenizer, text_tokenizer = load_tokenizers(cfg, device)
    model, phn2num = load_model(cfg, device)

    lip_video_path = resolve_lip_video_path(cfg)
    save_dir = os.path.join(cfg.result_dir, os.path.basename(os.path.dirname(lip_video_path)))
    audio_tokens, text_tokens, target_video = process_inputs(
        audio_tokenizer,
        text_tokenizer,
        phn2num,
        cfg,
        lip_video_path,
        save_dir,
        device,
    )

    inference_cfg = {
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "temperature": cfg.temperature,
        "stop_repetition": cfg.stop_repetition,
        "kvcache": cfg.kvcache,
        "silence_tokens": cfg.silence_tokens,
        "sample_batch_size": cfg.sample_batch_size,
    }

    concat_frames, gen_frames = inference_dubbing_sample(
        model,
        audio_tokens,
        text_tokens,
        target_video,
        device=device,
        decode_config=inference_cfg,
    )

    last_gen_frames = gen_frames[-1:, :, :]
    full_video_path = resolve_full_video_path(cfg)
    save_results(
        cfg.result_dir,
        lip_video_path,
        last_gen_frames,
        audio_tokenizer,
        full_video_path,
        lip_video_path,
        device,
    )


if __name__ == "__main__":
    main()
