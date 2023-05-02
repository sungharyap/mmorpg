from PIL import Image
import torch
from torchvision.utils import save_image
from models.map_model import map_model

model, tokenizer = map_model("baseline")

text = "a red car parked on the street"

encoded_text = tokenizer.encode_plus(text, padding="max_length", max_length=32, return_tensors="pt")

num_images = 1


noise = torch.randn(num_images, 100, 1, 1)

with torch.no_grad():
    generated_images = model(encoded_text["input_ids"], noise)
    print(generated_images.size())

for i in range(num_images):
    image = generated_images[i].cpu()
    image = (image + 1) / 2.0
    save_image(image, f"generated_{i}.png")