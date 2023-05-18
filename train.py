import torch
import torch.nn as nn
from torch import optim

from utils import get_data, split_dataset
from models.map_model import map_model

DATA_PATH = '/path/to/data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, tokenizer = map_model("baseline")

category, text, imgs = get_data(DATA_PATH, saved=True)
train_loader, test_loader = split_dataset(model, tokenizer, category, text, imgs, saved=True)


model.to(DEVICE)

epoch = 100
batch_size=64
lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

noise = torch.randn(len(train_loader.dataset), 100, 1, 1).to(DEVICE)
model.train(train_loader, noise, criterion, optimizer, epoch, batch_size, DEVICE)

noise = torch.randn(len(test_loader.dataset), 100, 1, 1).to(DEVICE)
ssim_score = model.eval(test_loader, noise, DEVICE)

print("SSIM Score: ", ssim_score)
