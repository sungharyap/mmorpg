import torch
import torch.nn as nn

# Text to Image Model
class TextToImageBaseline(nn.Module):
    def __init__(self, encoder, mlp, decoder):
        super(TextToImageBaseline, self).__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.decoder = decoder
        
    def forward(self, text, noise):
        encoded_text = self.encoder(text).last_hidden_state[:,0]
        mlp_output = self.mlp(encoded_text)
        
        mlp_output = mlp_output.view(-1, 100, 1, 1)
        decoder_input = torch.sum(torch.stack([mlp_output, noise], dim=0), dim=0)
        generated_image = self.decoder(decoder_input)
        return generated_image
