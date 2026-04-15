import argparse
import os
from typing import Iterable, List

import numpy as np
import torch

from src.utils.utils import move_to_cuda


def patch_multiprocessing_lock():
    import multiprocessing
    import threading

    try:
        multiprocessing.Lock()
    except PermissionError:
        multiprocessing.Lock = threading.Lock


def build_video_transform(av_utils):
    return av_utils.Compose(
        [
            av_utils.Normalize(0.0, 255.0),
            av_utils.CenterCrop((88, 88)),
            av_utils.Normalize(0.421, 0.165),
        ]
    )


def iter_video_paths(input_path: str, ext: str) -> Iterable[str]:
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for name in sorted(files):
                if name.lower().endswith(ext):
                    yield os.path.join(root, name)
    else:
        if input_path.lower().endswith(ext):
            yield input_path


def resolve_output_path(video_path: str, input_root: str, output_root: str) -> str:
    rel_path = os.path.relpath(video_path, input_root)
    rel_base, _ = os.path.splitext(rel_path)
    return os.path.join(output_root, rel_base + ".npy")


def extract_one(video_path: str, transform, model, device: torch.device, av_utils) -> np.ndarray:
    frames = av_utils.load_video(video_path)
    frames = transform(frames)
    frames = frames.astype(np.float32)
    video = torch.from_numpy(frames).unsqueeze(0)  # [1, T, H, W]

    sample = {"source": {"audio": None, "video": video.unsqueeze(1)}}  # [1, 1, T, H, W]
    sample = move_to_cuda(sample, device=device)

    with torch.no_grad():
        features, _ = model.extract_finetune(**sample)

    features = features[0].detach().cpu().float().numpy()  # [T, 1024]
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AV-HuBERT features from lip videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True, help="Path to AV-HuBERT checkpoint")
    parser.add_argument("--input", required=True, help="Input video file or directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for .npy files")
    parser.add_argument("--ext", default=".mp4", help="Video file extension")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files if output .npy already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    patch_multiprocessing_lock()

    from src.modeling.modules.avhubert.avhubert_wrapper import load_avhubert_model
    import importlib

    av_utils = importlib.import_module("src.modeling.modules.avhubert.utils")

    ext = args.ext.lower()
    if not ext.startswith("."):
        ext = "." + ext

    input_path = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output_dir)

    if not os.path.exists(input_path):
        print(f"Input path not found: {input_path}")
        return 1

    if os.path.isdir(input_path):
        input_root = input_path
    else:
        input_root = os.path.dirname(input_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    os.makedirs(output_root, exist_ok=True)

    model = load_avhubert_model(args.ckpt_path, modalities=["audio", "video"], use_cuda=False)
    model = model.to(device)
    model.eval()

    transform = build_video_transform(av_utils)

    video_paths = list(iter_video_paths(input_path, ext))
    if not video_paths:
        print(f"No video files found under {input_path} with extension {ext}")
        return 1

    total = len(video_paths)
    success = 0
    failed: List[str] = []

    for idx, video_path in enumerate(video_paths, start=1):
        out_path = resolve_output_path(video_path, input_root, output_root)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if args.skip_existing and os.path.exists(out_path):
            continue

        try:
            feats = extract_one(video_path, transform, model, device, av_utils)
            np.save(out_path, feats)
            success += 1
        except Exception as exc:
            failed.append(video_path)
            print(f"[FAIL] {video_path}: {exc}")

        if idx % 50 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    print(f"Done. success={success}, failed={len(failed)}, output_dir={output_root}")
    if failed:
        print("Failed files:")
        for path in failed:
            print(path)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
