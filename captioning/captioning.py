from datasets import load_dataset
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import argparse
import random

load_dotenv()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

BLIP1_BATCH = 32

'''
The images used are from the sled-umich/Action-Effect dataset: Qiaozi Gao, Shaohua Yang, Joyce Chai, and Lucy Vanderwende. 2018. What Action Causes This? Towards Naive Physical Action-Effect Prediction. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 934–945, Melbourne, Australia. Association for Computational Linguistics.
'''

def get_sample_images():
    '''
    This function returns fixed examples from "sled-umich/Action-Effect" to test the various
    captioning models. 
    '''

    dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)

    images_to_caption = [
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][1]['positive_image_list'][1]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][1]['positive_image_list'][3]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][2]['positive_image_list'][4]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][2]['positive_image_list'][0]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][2]['positive_image_list'][1]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][13]['positive_image_list'][0]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][12]['positive_image_list'][4]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][14]['positive_image_list'][6]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][15]['positive_image_list'][8]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][17]['positive_image_list'][4]},
        {"label": "arrange+chairs", "type": "positive", "image": dataset['ActionEffect'][19]['positive_image_list'][1]},
    ]

    return images_to_caption

def get_all_images():
    '''
    Gets all images in the dataset and formats them for captioning
    '''
    
    dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)
    
    images_to_caption = []

    for category in dataset['ActionEffect']:
        for img in category['positive_image_list']:
            images_to_caption.append({
                'label': category['verb noun'],
                'image': img,
            })
    
    return images_to_caption


def load_blip():
    if device.type == "cpu":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
    return model, processor

def load_blip2():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", revision="51572668da0eb669e01a189dc22abe6088589a24", torch_dtype=torch.float16).to(device)
    return model, processor

def generate_all_captions(images_to_caption, version="blip2", function_to_use=0, try_all=False, save_file='./saved_captions/captions.txt'):

    assert(len(images_to_caption) != 0)

    print(f"Starting to load the {version} model")

    if version == "blip":
        model, processor = load_blip()
        print("Loaded blip successfully!")
    elif version == "blip2":
        assert(device.type == "cuda")
        model, processor = load_blip2()
        print("Loaded blip2 successfully!")
    else:
        print(f"Invalid model chosen: {version}")
        return

    ### TODO: Add any more captioning functions here

    def generate_caption_blip_v0(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption
      
    def generate_caption_blip_v1(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, text="An image of", return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption
      
    def generate_caption_blip_v2(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, text="Give a detailed description of the image and the spatial relationships in it. This image shows", return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=80)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption

    def generate_caption_blip2_v0(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=20)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption
    
    def generate_caption_blip2_v1(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, text="An image of", return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption
    
    def generate_caption_blip2_v2(images_to_caption):
        for image_data in images_to_caption:
            image = image_data["image"]
            inputs = processor(images=image, text="Give a detailed description of the image and the spatial relationships in it. This image shows", return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=80)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            image_data["caption"] = caption
        return images_to_caption
    
    ### Make sure to add any new function here!
    blip_functions = [generate_caption_blip_v0, generate_caption_blip_v1, generate_caption_blip_v2]
    blip2_functions = [generate_caption_blip2_v0, generate_caption_blip2_v1, generate_caption_blip2_v2]

    if not try_all:
        if version == "blip":
            caption_function = blip_functions[function_to_use]
        else:
            caption_function = blip2_functions[function_to_use]
        
        print(f'Using prompt function {function_to_use}')
        print()

        images_to_caption = caption_function(images_to_caption)

        clipped_captions = []
        
        # Get how much filler text to skip at the start of the caption
        if version == 'blip':
            match function_to_use:
                case 0:
                    skipby = 0
                case 1:
                    skipby = len("An image of ")
                case 2:
                    skipby = len("Give a detailed description of the image and the spatial relationships in it. This image shows")
        else:
            match function_to_use:
                case 0:
                    skipby = 0
                case 1 | 2:
                    skipby = len("An image of ")
                case 2:
                    skipby = len("Give a detailed description of the image and the spatial relationships in it. This image shows")
        
        # Print captions
        for entry in images_to_caption:
            print("CAPTION: ", entry["caption"])
            clipped_captions.append(entry["caption"][skipby:])
        
        if save_file != '':
            with open(save_file, 'w') as f:
                f.write('\n'.join(clipped_captions))
        
        return images_to_caption

    else:
        print("(INFO) Trying functions that use on blip")
        for i, caption_function in enumerate(blip_functions):
            print(f"(INFO) Version {i}")
            images_to_caption = caption_function(images_to_caption)
            for entry in images_to_caption:
                # display(entry["image"])
                print("CAPTION: ", entry["caption"])
        
        print("(INFO) Trying functions that use on blip2")
        for i, caption_function in enumerate(blip2_functions):
            print(f"(INFO) Version {i}")
            images_to_caption = caption_function(images_to_caption)
            for entry in images_to_caption:
                # display(entry["image"])
                print("CAPTION: ", entry["caption"])
    
        return None
    

def main(params):
    if params.debug:
        generate_all_captions(get_all_images(), try_all=True, save_file=params.save_file)
    else:
        generate_all_captions(get_all_images(), version=params.version, function_to_use=params.function_to_use, try_all=False, save_file=params.save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Caption sample images using blip or blip2; to use a specific image set, call generate_all_captions() directly")
    parser.add_argument("--debug", type=bool, default=False, help="Use this for testing all captioning functions")
    parser.add_argument("--version", type=str, default="blip2", help="Specify which model to use")
    parser.add_argument("--function_to_use", type=int, default=0, help="Specify which captioning function to use")
    parser.add_argument('--save_file', type=str, default='captions.txt', help='Which file to save the captions to')
 
    params, unknown = parser.parse_known_args()
    model = main(params)