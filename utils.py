from typing import Tuple, List
import numpy as np
import torch as th
import os
import tqdm
import json
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image


def preprocess_image(img) -> th.Tensor:
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img)
    return img


def get_data(path: str, saved: bool) -> Tuple[List[str], List[str], th.Tensor]:
    with open(os.path.join(path, 'data_small.json')) as f:
        obj = json.load(f)['data']
        
    category = []
    text = []
    if not saved:
        imgs = []
        for e in tqdm.tqdm(obj):
            category.append(e['category'])
            text.append(e['text'].split("[SEP]")[0])
            img = Image.open(os.path.join(path, 'img', e['img'])).convert('RGB')
            imgs.append(preprocess_image(img))
        imgs = th.stack(imgs, dim=0)
        #imgs_numpy = imgs.numpy()
        #np.save('imgs_tensor', imgs_numpy)
    else:
        for e in tqdm.tqdm(obj):
            category.append(e['category'])
            text.append(e['text'].split("[SEP]")[0])
        
        np_img = np.load('saved/imgs_tensor.npy')
        imgs = th.from_numpy(np_img)

    return category, text, imgs


def split_dataset(model, tokenizer, category, text, imgs, saved: bool):
    # not use category now
    if not saved:
        text_embs = []
        for e in tqdm.tqdm(text):
            encoded_text = tokenizer.encode_plus(e, padding="max_length", max_length=32, return_tensors="pt")
            text_emb = model.encode(encoded_text["input_ids"])
            text_embs.append(text_emb)
        text_embs = th.stack(text_embs, dim=0)
        #text_embs_numpy = text_embs.detach().numpy()
        #np.save('text_embs_tensor', text_embs_numpy)
    else:
        np_text = np.load('saved/text_embs_tensor.npy')
        text_embs = th.from_numpy(np_text)
    
    k = int(len(text_embs)*0.8)
    train_dataset = TensorDataset(text_embs[:k], imgs[:k])
    test_dataset = TensorDataset(text_embs[k:], imgs[k:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_loader, test_loader