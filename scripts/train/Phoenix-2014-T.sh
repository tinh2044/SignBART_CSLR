#!/bin/bash

CONDA_ENV_NAME="base"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate

BATCH_SIZE=8
EPOCHS=100
FINETUNE=""
DEVICE="cpu"
SEED=0
RESUME=""
START_EPOCH=0
EVAL_FLAG=""
TEST_ON_LAST_EPOCH="False"
NUM_WORKERS=0
PIN_MEM="--pin-mem"
CFG_PATH="configs/phoenix-2014t.yaml"



python -m main \
   --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --finetune "$FINETUNE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --resume "$RESUME" \
    --start_epoch "$START_EPOCH" \
    $EVAL_FLAG \
    --test_on_last_epoch "$TEST_ON_LAST_EPOCH" \
    --num_workers "$NUM_WORKERS" \
    $PIN_MEM \
    --cfg_path "$CFG_PATH" \
    # $LOG_ALL
conda deactivate
