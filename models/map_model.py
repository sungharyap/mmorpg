import torch
import torch.nn.functional as F

from .baseline import GANDecoder, MLP, TextToImageBaseline

def map_model(model_type):
    if model_type == "baseline":
        from transformers import ElectraModel, ElectraTokenizer

        #encoder = ElectraModel.from_pretrained("google/electra-small-discriminator")
        #tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
        
        encoder = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

        return TextToImageBaseline(encoder, MLP(256, 512, 100), GANDecoder(1, 100, 64, 3)), tokenizer
    
    if model_type == "baseline-clip":
        from transformers import ElectraModel, ElectraTokenizer

        #encoder = ElectraModel.from_pretrained("google/electra-small-discriminator")
        #tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
        
        encoder = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

        return TextToImageBaseline(encoder, MLP(38, 512, 100), GANDecoder(1, 100, 64, 3)), tokenizer

    if model_type in ["baseline-sd", "baseline-stable_diffusion"]:
        """
        Beware that this returns pipeline not torch module
        """
        from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    
        repo = "Bingsu/my-korean-stable-diffusion-v1-5"
        euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_config(repo, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(
            repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float32,
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        return pipe
