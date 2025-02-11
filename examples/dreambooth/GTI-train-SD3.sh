#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export PERSON="mrt"
export INSTANCE_DATA_DIR="/mnt/Data/mrt/$PERSON"
export INSTANCE_PROMPT="a photo of $PERSON"
export CLASS_PROMPT="an ultra realistic photo of a man"
export OUTPUT_DIR="${PERSON}-model"

# There is currently a bug in HuggingFace that only allows this script to run on a single GPU. Make sure to execute "accelerate config" again if the script fails.
accelerate launch train_dreambooth_lora_sd3.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
	--instance_data_dir="$INSTANCE_DATA_DIR" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=1024 \
    --train_batch_size=1 \
    --train_text_encoder \
    --gradient_accumulation_steps=1 \
    --optimizer="prodigy" \
    --learning_rate=1.0 \
    --text_encoder_lr=1.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=2500 \
    --rank=32 \
    --seed="0" \
	--push_to_hub