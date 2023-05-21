from PIL import Image
import torch
import numpy as np
from torchvision.utils import save_image
from models.map_model import map_model
from koclip import load_koclip
from pytorch_msssim import SSIM

################################
model_type = "baseline-sd"
prompt = "Korean_input_text"
################################

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if model_type == "baseline":
    model, tokenizer = map_model("baseline")
    model.load_state_dict(torch.load('saved/model_parameters.pt'))
    
    encoded_text = tokenizer.encode_plus(prompt, padding="max_length", max_length=32, return_tensors="pt")
    num_images = 1
    
    noise = torch.randn(num_images, 100, 1, 1)
    
    with torch.no_grad():
        text_emb = model.encode(encoded_text["input_ids"])
        generated_images = model(text_emb, noise)
        print(generated_images.size())
    
    for i in range(num_images):
        image = generated_images[i].cpu()
        image = (image + 1) / 2.0
        save_image(image, f"generated_{i}.png")
    
elif model_type == "baseline-clip":
    model, tokenizer = map_model("baseline")
    _, processor = load_koclip("koclip-base")
    model.load_state_dict(torch.load('saved/clip_model_parameters.pt'))

    num_images = 1
    noise = torch.randn(num_images, 100, 1, 1)
    
    with torch.no_grad():
        np_text = np.load('./saved/koclip_text.npy')
        np_text = np.expand_dims(np_text, axis=1)
        text_embs = torch.from_numpy(np_text).type(torch.float32)
        text_emb = text_embs[0]
        generated_images = model(text_emb, noise)
        print(generated_images.size())

    for i in range(num_images):
        image = generated_images[i].cpu()
        image = (image + 1) / 2.0
        save_image(image, f"generated_{i}.png")

elif model_type in ["baseline-sd", "baseline-stable_diffusion"]:
    model = map_model(model_type)
    seed = 23957
    generator = torch.Generator(DEVICE).manual_seed(seed)
    image = model(prompt, num_inference_steps=25, generator=generator).images[0]
    image.save('generated.png')
