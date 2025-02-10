import torch
from diffusers import StableDiffusion3Pipeline


base_model = "stabilityai/stable-diffusion-3-medium-diffusers"

person = "mrt"
fine_tuned_model = f"{person}-model"
save_to = f"{person}-picture.png"
prompt = f"a photo of {person} as an evil debtor"

# Load base model
pipe = StableDiffusion3Pipeline.from_pretrained(
    base_model,
    # num_inference_steps=20,
    guidance_scale=2.0,
    height=1024,
    width=1024,
    torch_dtype=torch.float16,
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights(fine_tuned_model)

image = pipe(prompt).images[0]
image.save(save_to)
