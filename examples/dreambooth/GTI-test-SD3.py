import os
import torch
from datetime import datetime
from diffusers import StableDiffusion3Pipeline

base_model = "stabilityai/stable-diffusion-3-medium-diffusers"

person = "mrt"
fine_tuned_model = f"{person}-model"
save_to = f"{person}-picture.png"
prompt = f"a photo of {person} as a clown"

# Load base model
pipe = StableDiffusion3Pipeline.from_pretrained(
    base_model,
    # num_inference_steps=20,
    guidance_scale=2.0,
    height=1024,
    width=1024,
    torch_dtype=torch.float16,
).to("cuda")

# Create directory for images
image_dir = f"{person}-images"
os.makedirs(image_dir, exist_ok=True)

# Load LoRA weights
pipe.load_lora_weights(fine_tuned_model)

# Generate the image
image = pipe(prompt).images[0]

# Save image with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_to = os.path.join(image_dir, f"{timestamp}.png")
image.save(save_to)
