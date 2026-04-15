import argparse
import logging
import os
import shutil


def parse_lip_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    if base.endswith("_origin"):
        return None, None
    if "_" not in base:
        return None, None
    video_id, utt = base.rsplit("_", 1)
    if not utt.isdigit():
        return None, None
    return video_id, utt


def list_original_utts(pretrain_dir, video_id, cache):
    if video_id in cache:
        return cache[video_id]
    orig_dir = os.path.join(pretrain_dir, video_id)
    utts = []
    if os.path.isdir(orig_dir):
        for name in os.listdir(orig_dir):
            if not name.endswith(".mp4"):
                continue
            stem = os.path.splitext(name)[0]
            if not stem.isdigit():
                continue
            utts.append((int(stem), os.path.join(orig_dir, name)))
    utts.sort(key=lambda item: item[0])
    cache[video_id] = utts
    return utts


def list_lip_utts(lip_video_dir, video_id, cache):
    if video_id in cache:
        return cache[video_id]
    utts = []
    prefix = f"{video_id}_"
    for name in os.listdir(lip_video_dir):
        if not name.startswith(prefix) or not name.endswith(".mp4"):
            continue
        stem = os.path.splitext(name)[0]
        if stem.endswith("_origin"):
            continue
        _, utt = parse_lip_name(stem + ".mp4")
        if utt is None:
            continue
        utts.append(int(utt))
    utts.sort()
    cache[video_id] = utts
    return utts


def resolve_original_video(pretrain_dir, lip_video_dir, video_id, utt, orig_cache, lip_cache):
    candidates = []
    int_utt = int(utt)
    if int_utt >= 50000:
        candidates.append(int_utt - 50000)
    candidates.append(int_utt)

    for cand in candidates:
        if cand <= 0:
            continue
        cand_name = f"{cand:05d}.mp4"
        cand_path = os.path.join(pretrain_dir, video_id, cand_name)
        if os.path.exists(cand_path):
            return cand_path

    direct_path = os.path.join(pretrain_dir, video_id, f"{utt}.mp4")
    if os.path.exists(direct_path):
        return direct_path

    orig_utts = list_original_utts(pretrain_dir, video_id, orig_cache)
    lip_utts = list_lip_utts(lip_video_dir, video_id, lip_cache)
    if not orig_utts or not lip_utts:
        return None
    try:
        idx = lip_utts.index(int_utt)
    except ValueError:
        return None
    if idx < len(orig_utts):
        return orig_utts[idx][1]
    return None


def restore_videos_from_file_list(
    source_root_dir, file_list, lip_video_dir, output_dir, dry_run=False
):
    total = 0
    restored = 0
    missing_origin = 0
    missing_lip = 0
    skipped = 0

    with open(file_list, "r") as f:
        entries = [line.strip() for line in f if line.strip()]

    for entry in entries:
        parts = entry.split("/")
        if len(parts) < 2:
            continue
        video_id = parts[-2]
        utt = parts[-1]
        if not utt.isdigit():
            continue

        total += 1
        lip_name = f"{video_id}_{utt}.mp4"
        lip_path = os.path.join(lip_video_dir, lip_name)
        out_name = f"{video_id}_{utt}.mp4"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            skipped += 1
            continue
        if not os.path.exists(lip_path):
            missing_lip += 1
            logging.warning("Missing lip video for %s", lip_name)
            continue
        if os.path.isabs(entry):
            src_path = entry + ".mp4"
        else:
            src_path = os.path.join(source_root_dir, entry + ".mp4")
        if not os.path.exists(src_path):
            missing_origin += 1
            logging.warning("Missing original video for %s", entry)
            continue

        if dry_run:
            logging.info("Would copy %s -> %s", src_path, out_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(src_path, out_path)
        restored += 1

    logging.info(
        "Done. total=%d restored=%d skipped=%d missing_origin=%d missing_lip=%d",
        total,
        restored,
        skipped,
        missing_origin,
        missing_lip,
    )


def restore_videos(
    pretrain_dir,
    lip_video_dir,
    output_dir,
    file_list=None,
    source_root_dir=None,
    dry_run=False,
):
    if file_list:
        return restore_videos_from_file_list(
            source_root_dir,
            file_list,
            lip_video_dir,
            output_dir,
            dry_run=dry_run,
        )

    total = 0
    restored = 0
    missing = 0
    skipped = 0

    orig_cache = {}
    lip_cache = {}

    for name in sorted(os.listdir(lip_video_dir)):
        if not name.endswith(".mp4"):
            continue
        video_id, utt = parse_lip_name(name)
        if video_id is None:
            continue

        total += 1
        out_name = f"{video_id}_{utt}.mp4"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            skipped += 1
            continue

        orig_path = resolve_original_video(
            pretrain_dir,
            lip_video_dir,
            video_id,
            utt,
            orig_cache,
            lip_cache,
        )
        if orig_path is None:
            missing += 1
            logging.warning("Missing original video for %s", name)
            continue

        if dry_run:
            logging.info("Would copy %s -> %s", orig_path, out_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(orig_path, out_path)
        restored += 1

    logging.info(
        "Done. total=%d restored=%d skipped=%d missing=%d",
        total,
        restored,
        skipped,
        missing,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Restore original videos into trainval_preprocess/video with _origin suffix."
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        required=False,
        help="Path to pretrain directory containing original videos.",
    )
    parser.add_argument(
        "--source_root_dir",
        type=str,
        required=False,
        help="Root directory used when entries in file_list are relative.",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        required=False,
        help="file.list generated by save_wav; used for exact mapping.",
    )
    parser.add_argument(
        "--lip_video_dir",
        type=str,
        required=True,
        help="Path to trainval_preprocess/video with lip videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for original videos.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without copying files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.file_list and not args.source_root_dir:
        raise ValueError("--source_root_dir is required when --file_list is provided.")

    restore_videos(
        args.pretrain_dir,
        args.lip_video_dir,
        args.output_dir,
        file_list=args.file_list,
        source_root_dir=args.source_root_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
