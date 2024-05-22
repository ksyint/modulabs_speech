SAVE_PATH='./output/preprocess/pretrain_landmarks_and_bbox'
VIDEO_PATH="/app/lrs3/trainval"

HYDRA_FULL_ERROR=1 python ./src/core/preprocess/landmarks_detection/main.py \
save_path="$SAVE_PATH" \
Trainer.devices=\"0,\" \
module=baseline_lm_rotated_bbox \
module.backbone.device="cuda" \
module.save_path="$SAVE_PATH" \
module.dataset.video_path="$VIDEO_PATH"