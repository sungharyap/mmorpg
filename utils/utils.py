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


def get_data(path: str, saved) -> Tuple[List[str], List[str], th.Tensor]:
    with open(os.path.join(path, 'data_small.json')) as f:
        obj = json.load(f)['data']
    
    category = []
    text = []
    
    for e in tqdm.tqdm(obj):
        category.append(e['category'])
        text.append(e['text'].split("[SEP]")[0])
    
    if not saved:
        imgs = []
        for e in tqdm.tqdm(obj):
            img = Image.open(os.path.join(path, 'img', e['img'])).convert('RGB')
            imgs.append(preprocess_image(img))
        
        imgs = th.stack(imgs, dim=0)
        img_tensor = imgs.detach().cpu().numpy()
        np.save('img_tensor', img_tensor)
    
    np_img = np.load('./saved/img_tensor.npy')
    imgs = th.from_numpy(np_img)
    
    return category, text, imgs


def split_dataset(model, tokenizer, category, text, imgs, model_type, saved):
    # not use category now..
    if not saved:
        if model_type == "baseline":
            text_embs = []
        
            for e in tqdm.tqdm(text):
                encoded_text = tokenizer.encode_plus(e, padding="max_length", max_length=32, return_tensors="pt")
                text_emb = model.encode(encoded_text["input_ids"])
                text_emb = text_emb.detach().cpu().numpy()
                text_embs.append(text_emb)
                
            text_embs = th.stack(text_embs, dim=0)
            text_tensor = text_embs.detach().cpu().numpy()
            np.save('text_tensor', text_tensor)
        
    if model_type == "baseline":
        np_text = np.load('./saved/text_tensor.npy')
        text_embs = th.from_numpy(np_text)
        
    elif model_type == "baseline-clip":
        np_text = np.load('./saved/koclip_text_tensor.npy')
        np_text = np.expand_dims(np_text, axis=1)
        text_embs = th.from_numpy(np_text).type(th.float32)
    
    k = int(len(text_embs)*0.8)
    train_dataset = TensorDataset(text_embs[:k], imgs[:k])
    test_dataset = TensorDataset(text_embs[k:], imgs[k:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_loader, test_loader