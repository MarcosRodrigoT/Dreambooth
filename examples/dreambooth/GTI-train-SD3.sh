#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export PERSON="mrt"
export INSTANCE_DATA_DIR="/mnt/Data/mrt/mrt"
export INSTANCE_PROMPT="a photo of $PERSON"
export VALIDATION_PROMPT="a photo of $PERSON wearing a hat"
export CLASS_PROMPT="an ultra realistic photo of a man"
export OUTPUT_DIR="${PERSON}-model"

accelerate launch train_dreambooth_lora_sd3.py \
	--pretrained_model_name_or_path="$MODEL_NAME" \
	--gradient_checkpointing \
	--use_8bit_adam \
	--instance_data_dir="$INSTANCE_DATA_DIR" \
	--output_dir="$OUTPUT_DIR" \
	--mixed_precision="fp16" \
	--instance_prompt="$INSTANCE_PROMPT" \
	--class_prompt="$CLASS_PROMPT" \
	--resolution=1024 \
	--train_batch_size=1 \
	--gradient_accumulation_steps=4 \
	--learning_rate=1e-4 \
	--lr_scheduler="constant" \
	--lr_warmup_steps=0 \
	--max_train_steps=500 \
	--seed="0" \
	--push_to_hub