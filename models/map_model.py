import torch.nn.functional as F

from .baseline import GANDecoder, MLP, TextToImageBaseline

def map_model(model_type):
    if model_type == "baseline":
        from transformers import ElectraModel, ElectraTokenizer

        encoder = ElectraModel.from_pretrained("google/electra-small-discriminator")
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

        return TextToImageBaseline(encoder, MLP(256, 512, 100), GANDecoder(1, 100, 64, 3)), tokenizer
