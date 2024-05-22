SAVE_PATH='./output/unsupervised/PCL'

HYDRA_FULL_ERROR=1 python src/core/train/unsupervised/codebook/main.py \
save_path="$SAVE_PATH" \
module.batch_size=10 \
Trainer.devices=\"3,4,5\" \
Trainer.accumulate_grad_batches=1 \
module.dataset.train.json_path="/app/Talking_Head_Generation_2/preprocess_log/Lrs3_train_valid.json" \
module.dataset.train.dir_path="/app/lrs3/trainval" \
module.dataset.validation.json_path="/app/Talking_Head_Generation_2/preprocess_log/Lrs3_train_valid.json" \
module.dataset.validation.dir_path="/app/lrs3/trainval" \
module.dataset.test.dir_path="/app/lrs3/test"
