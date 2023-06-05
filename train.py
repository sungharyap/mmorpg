import torch
import torch.nn as nn
from torch import optim

from utils.utils import get_data, split_dataset
from models.map_model import map_model

DATA_PATH = './../mmorpg_data/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_type = "baseline-clip"

model, tokenizer = map_model(model_type)


category, text, imgs = get_data(DATA_PATH, saved=True)
train_loader, test_loader = split_dataset(model, tokenizer, category, text, imgs, model_type, saved=True)

model.to(DEVICE)

epoch = 2000
batch_size=64
lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

noise = torch.randn(len(train_loader.dataset), 100, 1, 1).to(DEVICE)
model.train(train_loader, noise, criterion, optimizer, epoch, batch_size, DEVICE)

model.load_state_dict(torch.load('saved/clip_model_parameters.pt'))

noise = torch.randn(len(test_loader.dataset), 100, 1, 1).to(DEVICE)
ssim = model.eval(test_loader, noise, DEVICE)
print(ssim)