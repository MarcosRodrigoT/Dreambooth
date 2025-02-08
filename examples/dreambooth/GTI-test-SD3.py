import torch
from diffusers import StableDiffusion3Pipeline


model_id = "mrt-model"
save_to = "mrt-picture.png"
prompt = "a photo of a clown"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    num_inference_steps=20,
    guidance_scale=1.0,
    height=1024,
    width=1024,
    torch_dtype=torch.float16,
).to("cuda")


image = pipe(prompt).images[0]
image.save(save_to)

