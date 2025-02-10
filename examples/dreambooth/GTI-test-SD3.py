import os
import torch
from datetime import datetime
from diffusers import StableDiffusion3Pipeline

base_model = "stabilityai/stable-diffusion-3-medium-diffusers"

PERSON = "mrt"
FINE_TUNED_MODEL = f"{PERSON}-model"
SAVE_TO = f"{PERSON}-picture.png"
PROMPT = f"a photo of {PERSON} as a clown"
NEGATIVE_PROMPT = "blurry, low quality, low resolution"

# Load base model
pipe = StableDiffusion3Pipeline.from_pretrained(
    base_model,
    # num_inference_steps=20,
    guidance_scale=5.0,
    height=1024,
    width=1024,
    torch_dtype=torch.float16,
).to("cuda")

# Create directory for images
image_dir = f"{PERSON}-images"
os.makedirs(image_dir, exist_ok=True)

# Load LoRA weights
pipe.load_lora_weights(FINE_TUNED_MODEL)

# Generate the image
image = pipe(
    PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
).images[0]

# Save image with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
SAVE_TO = os.path.join(image_dir, f"{timestamp}.png")
image.save(SAVE_TO)
