#!/bin/bash

export PERSON="mrt"
export INSTANCE_DATA_DIR="/mnt/Data/mrt/mrt"
export INSTANCE_PROMPT="a photo of $PERSON"
export CLASS_PROMPT="an ultra realistic photo of a man"
export OUTPUT_DIR="${PERSON}-model"

accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --mixed_precision="fp16" \
        --gradient_checkpointing \
        --use_8bit_adam \
        --instance_data_dir="$INSTANCE_DATA_DIR" \
        --instance_prompt="$INSTANCE_PROMPT" \
        --resolution=512 \
        --train_batch_size=1 \
        --class_prompt="$CLASS_PROMPT" \
        --gradient_accumulation_steps=1 \
        --train_text_encoder \
        --learning_rate=5e-6 \
        --num_train_epochs=100 \
        --output_dir="$OUTPUT_DIR" \
        --push_to_hub
