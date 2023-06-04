import torch
import torch.nn as nn
from pytorch_msssim import SSIM


# Text to Image Model
class TextToImageBaseline(nn.Module):
    def __init__(self, encoder, mlp, decoder):
        super(TextToImageBaseline, self).__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.decoder = decoder
        
    def forward(self, text_emb, noise):
        mlp_output = self.mlp(text_emb)
        
        mlp_output = mlp_output.view(-1, 100, 1, 1)
        decoder_input = torch.sum(torch.stack([mlp_output, noise], dim=0), dim=0)
        generated_image = self.decoder(decoder_input)
        return generated_image
    
    def encode(self, text):
        return self.encoder(text).last_hidden_state[:,0]

    def train(self, train_loader, noise, criterion, optimizer, epoch, batch_size, device):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        enabled = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        noise.requires_grad = False
        
        for i in range(epoch):
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                batch_noise = noise[batch_idx:batch_idx+len(X)]
                pred = self.forward(X, batch_noise)
                loss = criterion(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    print('Epoch: %d, [%5d/%5d] loss: %.6f' % (i+1, batch_idx*len(X), len(train_loader.dataset), loss))
                    
        torch.save(self.state_dict(), 'saved/model_parameters.pt')
    
    def eval(self, test_loader, noise, device):
        loss = []
        ssim_loss = SSIM(win_size=1, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)
                batch_noise = noise[batch_idx:batch_idx+len(X)]
                pred = self.forward(X, batch_noise)
                
                y = (y + 1) / 2
                pred = (pred + 1) / 2
                
                loss.append(ssim_loss(y, pred))
        
        return sum(loss)/len(loss)