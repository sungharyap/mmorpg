from PIL import Image
import torch
from torchvision.utils import save_image
from models.map_model import map_model
from pytorch_msssim import SSIM

model, tokenizer = map_model("baseline")
model.load_state_dict(torch.load('saved/model_parameters.pt'))
text = "Korean_input_text"

encoded_text = tokenizer.encode_plus(text, padding="max_length", max_length=32, return_tensors="pt")

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
