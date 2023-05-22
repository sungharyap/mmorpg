import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from pytorch_msssim import SSIM
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.utils import save_image

from models.map_model import map_model

# Load the inception model
inception_model = inception_v3(pretrained=True)
inception_model = inception_model.eval()

# Resize the image and normalize it according to inception's specifications
transform_input = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_images(directory):
    for file in glob.glob(directory):
        yield Image.open(file)

def get_features(img_iterator, count=5000):
    all_features = []
    for i, img in enumerate(img_iterator):
        if i >= count:
            break
        image_tensor = transform_input(img).unsqueeze(0).cuda()
        features = inception_model(image_tensor).detach().cpu().numpy()
        all_features.append(features)
    return np.concatenate(all_features)

def calculate_fid(real_dir: str, fake_dir: str, count=5000):
    """
    Example::
        >>> fid = calculate_fid('real/*.jpg', 'fake/*.jpg')
        >>> print('FID Score: ', fid)
    """

    real_features = get_features(get_images(real_dir), count)
    fake_features = get_features(get_images(fake_dir), count)

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu_real - mu_fake)**2.0)
    cov_mean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0*cov_mean)
    return fid

################################
model_type = "baseline-sd"
prompt = "Korean_input_text"
################################

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = "cpu"

dataset = load_dataset("jeongwoo25/mmorpg_world", data_files="train.zip")

model = map_model(model_type)
seed = 23957
generator = torch.Generator(DEVICE).manual_seed(seed)
dataset = dataset["train"][:16]

# for i, img in enumerate(dataset["image"]):
#     img.save(f"outputs/input_{i}.png")

images = model(dataset["text"], num_inference_steps=25, generator=generator).images
for i, img in enumerate(images):
    img.save(f"outputs/output_{i}.png")
