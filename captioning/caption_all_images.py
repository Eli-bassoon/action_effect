from load_captioning import *
from load_dataset import *
import pickle

output = {
    'verb-noun': [],
    'captions': [],
}

for category in dataset['ActionEffect']:
    print('Captioning', category['verb noun'])
    captions = caption_images(category['positive_image_list'])
    
    output['verb-noun'].extend([category['verb noun']] * len(captions))
    output['captions'].extend(captions)

with open('blip-large-captions.pt', 'wb') as f:
    pickle.dump(output, f)
