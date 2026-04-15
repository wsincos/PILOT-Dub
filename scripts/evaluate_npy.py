import json
import logging
import os
import sys
import time

import hydra
import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.inference import (  # noqa: E402
    build_text_tokens_for_model,
    inference_dubbing_sample,
    load_model,
    load_tokenizers,
    patch_multiprocessing_lock,
    replace_audio_in_video,
    resolve_device,
    seed_everything,
)
from src.data.tokenizer import tokenize_audio  # noqa: E402


def setup_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_list(list_path):
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 6:
                logging.warning("Line %d has %d columns: %s", line_idx, len(parts), line)
                continue
            if len(parts) == 6:
                speaker, target_id, target_text, target_avhubert_num, ref_id, ref_text = parts
                ref_speaker = speaker
            else:
                if len(parts) > 7:
                    parts = parts[:6] + [" | ".join(parts[6:])]
                speaker, target_id, target_text, target_avhubert_num, ref_speaker, ref_id, ref_text = parts
            entries.append(
                {
                    "line_idx": line_idx,
                    "speaker": speaker,
                    "target_id": target_id,
                    "target_text": target_text,
                    "target_avhubert_num": target_avhubert_num,
                    "ref_speaker": ref_speaker,
                    "ref_id": ref_id,
                    "ref_text": ref_text,
                }
            )
    return entries


def resolve_media_path(base_dir, speaker, file_id, extensions):
    for ext in extensions:
        candidate = os.path.join(base_dir, speaker, f"{file_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_audio_from_template(base_dir, template, **fields):
    if not template:
        return None
    candidate = os.path.join(base_dir, template.format(**fields))
    if os.path.exists(candidate):
        return candidate
    return None


def resolve_feature_path(npy_dir, speaker, target_id, target_avhubert_num, template, use_target_avhubert_num):
    if use_target_avhubert_num and target_avhubert_num and target_avhubert_num != "_":
        feature_id = target_avhubert_num
    else:
        feature_id = target_id
    filename = template.format(speaker=speaker, target_id=feature_id)
    candidate = os.path.join(npy_dir, filename)
    if os.path.exists(candidate):
        return candidate
    return None


def _resolve_lip_path(target_video_path, suffix):
    base, ext = os.path.splitext(target_video_path)
    if not suffix:
        return f"{base}_lip{ext}"
    if suffix.endswith(ext):
        affix = suffix[:-len(ext)]
    else:
        affix = suffix
    if not affix.startswith("_") and not affix.startswith("."):
        affix = f"_{affix}"
    return f"{base}{affix}{ext}"


def resolve_existing_lip_video(target_video_path, preferred_suffix):
    preferred_path = _resolve_lip_path(target_video_path, preferred_suffix)
    if preferred_suffix and os.path.exists(preferred_path):
        return preferred_path
    default_path = _resolve_lip_path(target_video_path, None)
    if os.path.exists(default_path):
        return default_path
    return None


def save_outputs(
    save_dir,
    gen_audio,
    audio_tokenizer,
    full_video_path,
    lip_video_path,
    device,
    output_basename,
):
    os.makedirs(save_dir, exist_ok=True)
    for stale_name in ("gen_0.mp4", "gen_0_lip.mp4", "gen_0.wav"):
        stale_path = os.path.join(save_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    gen = gen_audio.to(device)
    if gen.ndim == 2:
        gen = gen.unsqueeze(0)
    decoded = audio_tokenizer.decode([(gen, None)])
    wav_path = os.path.join(save_dir, "gen_0.wav")
    torchaudio.save(wav_path, decoded[0].cpu(), 16000)

    if full_video_path is not None:
        replace_audio_in_video(full_video_path, wav_path, os.path.join(save_dir, "gen_0.mp4"))
    if lip_video_path is not None:
        replace_audio_in_video(lip_video_path, wav_path, os.path.join(save_dir, "gen_0_lip.mp4"))

    output_mp4 = os.path.join(save_dir, "gen_0.mp4")
    output_lip_mp4 = os.path.join(save_dir, "gen_0_lip.mp4")
    output_wav = os.path.join(save_dir, "gen_0.wav")
    if os.path.exists(output_mp4):
        os.replace(output_mp4, os.path.join(save_dir, f"{output_basename}.mp4"))
    if lip_video_path is not None and os.path.exists(output_lip_mp4):
        os.replace(output_lip_mp4, os.path.join(save_dir, f"{output_basename}_lip.mp4"))
    if os.path.exists(output_wav):
        os.replace(output_wav, os.path.join(save_dir, f"{output_basename}.wav"))


def write_summary(summary_path, record):
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


@hydra.main(config_path="../configs", config_name="evaluate/LRS3_Test_True", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)
    patch_multiprocessing_lock()

    if bool(OmegaConf.select(cfg, "mask_avhubert", default=False)):
        os.environ["MASK_AVHUBERT"] = "1"
    else:
        os.environ.pop("MASK_AVHUBERT", None)

    device = resolve_device(cfg)
    if "avhubert_model" in cfg.model and "use_cuda" in cfg.model.avhubert_model:
        cfg.model.avhubert_model.use_cuda = device.startswith("cuda")

    os.makedirs(cfg.result_root, exist_ok=True)
    logger = setup_logging(os.path.join(cfg.result_root, "evaluate_npy.log"))

    logger.info("Loading model and tokenizers.")
    audio_tokenizer, text_tokenizer = load_tokenizers(cfg, device)
    model, phn2num = load_model(cfg, device)

    entries = parse_list(cfg.list_path)
    if not entries:
        logger.warning("No entries found in list: %s", cfg.list_path)
        return

    decode_config = {
        "top_k": cfg.top_k,
        "top_p": cfg.top_p,
        "temperature": cfg.temperature,
        "stop_repetition": cfg.stop_repetition,
        "kvcache": cfg.kvcache,
        "silence_tokens": cfg.silence_tokens,
        "sample_batch_size": cfg.sample_batch_size,
    }

    max_items = int(OmegaConf.select(cfg, "max_items", default=-1))
    min_frame = int(OmegaConf.select(cfg, "min_frame", default=-1))
    skip_existing = bool(OmegaConf.select(cfg, "skip_existing", default=True))

    systems = OmegaConf.select(cfg, "system_names", default=None)
    if systems is None:
        systems = OmegaConf.select(cfg, "system_name", default="system1")
    if isinstance(systems, str):
        systems = [s.strip() for s in systems.split(",") if s.strip()]
    if not systems:
        logger.error("No system_names specified.")
        return

    for system_name in systems:
        if system_name == "system1":
            reference_mode = "target"
        elif system_name == "system2":
            reference_mode = "ref"
        else:
            logger.error("Unknown system name: %s", system_name)
            continue

        result_dir = os.path.join(cfg.result_root, system_name)
        os.makedirs(result_dir, exist_ok=True)
        logger = setup_logging(os.path.join(result_dir, "run.log"))

        summary_path = os.path.join(result_dir, "summary.jsonl")
        if not bool(OmegaConf.select(cfg, "append_summary", default=False)):
            if os.path.exists(summary_path):
                os.remove(summary_path)

        total = 0
        skipped = 0
        failed = 0
        succeeded = 0
        start_time = time.time()

        logger.info("Starting evaluation for %d entries.", len(entries))

        progress = entries
        if tqdm is not None:
            progress = tqdm(entries, desc=f"eval-{system_name}", unit="utt")

        for entry in progress:
            if max_items > 0 and total >= max_items:
                break
            total += 1

            speaker = entry["speaker"]
            target_id = entry["target_id"]
            target_text = entry["target_text"]
            target_avhubert_num = entry["target_avhubert_num"]
            ref_id = entry["ref_id"]
            ref_text = entry["ref_text"]

            if reference_mode == "target":
                src_id = target_id
                src_text = target_text
            elif reference_mode == "ref":
                src_id = ref_id
                src_text = ref_text
            else:
                logger.error("Unknown reference_mode: %s", reference_mode)
                break

            save_dir = os.path.join(result_dir, speaker)
            output_wav = os.path.join(save_dir, f"{target_id}.wav")
            output_mp4 = os.path.join(save_dir, f"{target_id}.mp4")
            output_lip_mp4 = os.path.join(save_dir, f"{target_id}_lip.mp4")
            output_wav_only = bool(OmegaConf.select(cfg, "output_wav_only", default=False))
            if skip_existing:
                if output_wav_only and os.path.exists(output_wav):
                    skipped += 1
                    write_summary(
                        summary_path,
                        {
                            "speaker": speaker,
                            "target_id": target_id,
                            "system_name": system_name,
                            "status": "skipped",
                            "output_dir": save_dir,
                        },
                    )
                    continue
                if (not output_wav_only) and os.path.exists(output_mp4) and (
                    not OmegaConf.select(cfg, "require_lip_output", default=False) or os.path.exists(output_lip_mp4)
                ):
                    skipped += 1
                    write_summary(
                        summary_path,
                        {
                            "speaker": speaker,
                            "target_id": target_id,
                            "system_name": system_name,
                            "status": "skipped",
                            "output_dir": save_dir,
                        },
                    )
                    continue

            target_video_path = None
            if not output_wav_only:
                target_video_path = resolve_media_path(
                    cfg.test_dir, speaker, target_id, cfg.target_video_exts
                )
                if not target_video_path:
                    failed += 1
                    expected_targets = [
                        os.path.join(cfg.test_dir, speaker, f"{target_id}{ext}")
                        for ext in cfg.target_video_exts
                    ]
                    logger.warning(
                        "Missing target video for %s/%s. Expected one of: %s",
                        speaker,
                        target_id,
                        expected_targets,
                    )
                    write_summary(
                        summary_path,
                        {
                            "speaker": speaker,
                            "target_id": target_id,
                            "system_name": system_name,
                            "status": "missing_target_video",
                            "expected_paths": expected_targets,
                        },
                    )
                    continue

            target_audio_template = OmegaConf.select(cfg, "target_audio_template", default="")
            reference_audio_template = OmegaConf.select(cfg, "reference_audio_template", default="")
            expected_refs = []
            if reference_mode == "target":
                if target_audio_template:
                    template_path = os.path.join(
                        cfg.test_dir, target_audio_template.format(speaker=speaker, target_id=target_id)
                    )
                    expected_refs.append(template_path)
                reference_audio_path = resolve_audio_from_template(
                    cfg.test_dir,
                    target_audio_template,
                    speaker=speaker,
                    target_id=target_id,
                )
                if not reference_audio_path:
                    reference_audio_path = resolve_media_path(
                        cfg.test_dir, speaker, target_id, cfg.reference_audio_exts
                    )
                expected_refs += [
                    os.path.join(cfg.test_dir, speaker, f"{target_id}{ext}")
                    for ext in cfg.reference_audio_exts
                ]
            else:
                ref_speaker = entry.get("ref_speaker", speaker)
                if reference_audio_template:
                    template_path = os.path.join(
                        cfg.test_dir,
                        reference_audio_template.format(
                            ref_speaker=ref_speaker,
                            ref_id=src_id,
                            speaker=speaker,
                            target_id=target_id,
                        ),
                    )
                    expected_refs.append(template_path)
                reference_audio_path = resolve_audio_from_template(
                    cfg.test_dir,
                    reference_audio_template,
                    speaker=speaker,
                    target_id=target_id,
                    ref_speaker=ref_speaker,
                    ref_id=src_id,
                )
                if not reference_audio_path:
                    reference_audio_path = resolve_media_path(
                        cfg.test_dir, ref_speaker, src_id, cfg.reference_audio_exts
                    )
                expected_refs += [
                    os.path.join(cfg.test_dir, ref_speaker, f"{src_id}{ext}")
                    for ext in cfg.reference_audio_exts
                ]
            if not reference_audio_path:
                failed += 1
                logger.warning(
                    "Missing reference audio for %s/%s. Expected one of: %s",
                    speaker,
                    src_id,
                    expected_refs,
                )
                write_summary(
                    summary_path,
                    {
                        "speaker": speaker,
                        "target_id": target_id,
                        "system_name": system_name,
                        "status": "missing_reference_audio",
                        "expected_paths": expected_refs,
                    },
                )
                continue

            feature_path = resolve_feature_path(
                cfg.npy_dir,
                speaker,
                target_id,
                target_avhubert_num,
                cfg.npy_filename_template,
                bool(OmegaConf.select(cfg, "use_target_avhubert_num", default=False)),
            )
            if not feature_path:
                failed += 1
                expected = os.path.join(
                    cfg.npy_dir,
                    cfg.npy_filename_template.format(speaker=speaker, target_id=target_id),
                )
                logger.warning("Missing npy feature for %s/%s. Expected: %s", speaker, target_id, expected)
                write_summary(
                    summary_path,
                    {
                        "speaker": speaker,
                        "target_id": target_id,
                        "system_name": system_name,
                        "status": "missing_npy_feature",
                        "expected_path": expected,
                    },
                )
                continue

            text_tokens = build_text_tokens_for_model(
                text_tokenizer,
                phn2num,
                src_text,
                target_text,
                cfg,
            )
            if text_tokens is None:
                failed += 1
                logger.warning("Empty text tokens for %s/%s", speaker, target_id)
                write_summary(
                    summary_path,
                    {
                        "speaker": speaker,
                        "target_id": target_id,
                        "system_name": system_name,
                        "status": "empty_text_tokens",
                    },
                )
                continue

            record = {
                "speaker": speaker,
                "target_id": target_id,
                "system_name": system_name,
                "reference_id": src_id,
                "reference_mode": reference_mode,
                "output_dir": save_dir,
                "target_video_path": target_video_path,
                "reference_audio_path": reference_audio_path,
                "feature_path": feature_path,
            }

            try:
                encoded_frames = tokenize_audio(audio_tokenizer, reference_audio_path, offset=0)
                audio_tokens = encoded_frames[0][0]
                if audio_tokens.ndim == 2:
                    audio_tokens = audio_tokens.unsqueeze(0)

                feature = np.load(feature_path)
                if feature.ndim != 2:
                    raise ValueError(f"Expected npy shape [T, C], got {feature.shape}")
                # Pass [T, C] so prepare_generate_inputs can add batch dim.
                v = torch.from_numpy(feature).float()

                lip_video_path = None
                if target_video_path is not None:
                    lip_video_path = resolve_existing_lip_video(
                        target_video_path,
                        OmegaConf.select(cfg, "lip_suffix", default=None),
                    )

                if min_frame >= 0 and v.shape[0] < min_frame:
                    skipped += 1
                    logger.info("Skipping %s/%s (frames=%d)", speaker, target_id, v.shape[0])
                    record["status"] = "skipped_short_video"
                    write_summary(summary_path, record)
                    continue

                _, gen_frames = inference_dubbing_sample(
                    model,
                    audio_tokens,
                    text_tokens,
                    v,
                    device=device,
                    decode_config=decode_config,
                )
                last_gen_frames = gen_frames[-1:, :, :]

                save_outputs(
                    save_dir,
                    last_gen_frames,
                    audio_tokenizer,
                    target_video_path,
                    lip_video_path,
                    device,
                    output_basename=target_id,
                )

                succeeded += 1
                record["status"] = "ok"
                write_summary(summary_path, record)
            except Exception as exc:
                failed += 1
                logger.exception("Failed on %s/%s: %s", speaker, target_id, exc)
                record["status"] = "error"
                record["error"] = str(exc)
                record["error_type"] = exc.__class__.__name__
                write_summary(summary_path, record)

        elapsed = time.time() - start_time
        logger.info(
            "Done. total=%d success=%d skipped=%d failed=%d time=%.2fs",
            total,
            succeeded,
            skipped,
            failed,
            elapsed,
        )


if __name__ == "__main__":
    main()
