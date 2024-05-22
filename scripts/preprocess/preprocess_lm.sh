SAVE_PATH='./output/preprocess/pretrain_landmarks_and_bbox'
VIDEO_PATH='/app/lrs3/pretrain'

HYDRA_FULL_ERROR=1 python ./src/core/preprocess/landmarks_detection/main.py \
save_path="$SAVE_PATH" \
module.save_path="$SAVE_PATH" \
module.dataset.preprocess.video_path="$VIDEO_PATH"