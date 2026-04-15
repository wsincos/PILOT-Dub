#!/bin/bash

PROJECT_DIR=/data1/jinyu_wang/projects/VoiceCraft-Dub-Lightning
cd $PROJECT_DIR || exit
export PYTHONPATH=/data1/jinyu_wang/projects/VoiceCraft-Dub-Lightning:$PYTHONPATH

ENCODEC=${PROJECT_DIR}/artifacts/pretrained_models/tokenizers/encodec.th
SPLIT_NAME=trainval
ROOT_DIR_PATH=/data1/jinyu_wang/datasets/LRS3_Dataset/mp4 # Path to your dataset root directory
SAVE_DIR_PATH=/data1/jinyu_wang/datasets/LRS3_Dataset/mp4/${SPLIT_NAME}_preprocess # Path to save the preprocessed data
FACE_PREPROCESS=${PROJECT_DIR}/artifacts/pretrained_models/landmarks

# # #Extract wav file from video and save file.list
# echo "Start extracting wav..."
# python src/data/utils/save_wav.py \
#     --root_dir "$ROOT_DIR_PATH" \
#     --split_name "$SPLIT_NAME"

# ## Extract audio tokens and phoneme tokens
# echo "Start processing audio..."
# python src/data/utils/phonemize_lrs.py \
#     --save_dir "$SAVE_DIR_PATH" \
#     --root_dir "$ROOT_DIR_PATH" \
#     --encodec_model_path "$ENCODEC" \
#     --split_name "$SPLIT_NAME"

# ## Preprocess lip video
# echo "Start processing lip video..."
# python src/data/utils/detect_landmark.py --root ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark \
#  --manifest ${ROOT_DIR_PATH}/file.list --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}

# echo "Start aligning mouth..."
# python src/data/utils/align_mouth.py --video-direc ${ROOT_DIR_PATH} --landmark ${SAVE_DIR_PATH}/landmark --filename-path ${ROOT_DIR_PATH}/file.list \
#  --save-direc ${SAVE_DIR_PATH}/video  --ffmpeg ffmpeg --face_preprocess_dir ${FACE_PREPROCESS}

# #Make a continuation form for constructing training dataset
# echo "Start constructing dataset..."
# python src/data/utils/construct_dataset.py --root_dir ${SAVE_DIR_PATH}

# Restore full videos into video_origin (exact mapping by/ file.list)
# echo "Start restoring original videos..."
# python src/data/utils/restore_original_videos.py \
#     --source_root_dir "${ROOT_DIR_PATH}" \
#     --file_list "${ROOT_DIR_PATH}/file.list" \
#     --lip_video_dir "${SAVE_DIR_PATH}/video" \
#     --output_dir "${SAVE_DIR_PATH}/video_origin"

# Extract AV-HuBERT lip features and save as .npy
echo "Start extracting AV-HuBERT lip features..."
AVHUBERT_CKPT=${PROJECT_DIR}/artifacts/pretrained_models/large_lrs3_iter5.pt
python src/data/utils/extract_avhubert_features.py \
    --ckpt_path "${AVHUBERT_CKPT}" \
    --input "${SAVE_DIR_PATH}/video" \
    --output_dir "${SAVE_DIR_PATH}/lip_feature" \
    --ext .mp4 \
    --device cuda \
    --skip_existing

# Build target-side MFA phone alignments and frame labels for V9-style alignment supervision
# This is intentionally kept as a separate offline step and is NOT run by default.
# Recommended environment: voicecraft (contains local MFA models).
# Example:
# conda run -n voicecraft python src/data/utils/generate_mfa_alignment_labels.py \
#     --raw_root_dir "${ROOT_DIR_PATH}" \
#     --preprocess_dir "${SAVE_DIR_PATH}" \
#     --output_dir "${SAVE_DIR_PATH}/mfa_target_alignment" \
#     --splits train validation \
#     --skip_existing
