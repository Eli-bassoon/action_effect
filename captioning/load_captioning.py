import torch
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

capt_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", max_new_tokens=100)
capt_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

def caption_images(imgs):
    # unconditional image captioning
    inputs = capt_processor(imgs, return_tensors="pt").to("cuda", torch.float16)
    out = capt_model.generate(**inputs)
    
    if out.shape[0] > 1:
        return [capt_processor.decode(out_row, skip_special_tokens=True) for out_row in out]
    else:
        return capt_processor.decode(out[0], skip_special_tokens=True)
