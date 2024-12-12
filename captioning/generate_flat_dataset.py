import math
from datasets import load_from_disk, Dataset
import pickle
import os

flat_dataset = load_from_disk('./datasets/flat_dataset')

RUN_NAME = 'blip2-0'

with open(os.path.join('./saved_captions', RUN_NAME+'.pt'), 'rb') as f:
    captions = pickle.load(f)

train_split = 0.2
val_split = 0.1

train_captions = []
val_captions = []
test_captions = []

i = 0
for category in dataset['ActionEffect']:
    n = len(category['positive_image_list'])
    train_n = math.ceil(n*train_split)
    val_n = math.ceil(n*val_split)
    test_n = n - (train_n + val_n)
    
    train_captions.extend([e['caption'] for e in captions[i : i+train_n]])
    val_captions.extend([e['caption'] for e in captions[i+train_n : i+train_n+val_n]])
    test_captions.extend([e['caption'] for e in captions[i+train_n+val_n : i+n]])
    
    i += n

flat_dataset['train'] = flat_dataset['train'].add_column('captions', train_captions)
flat_dataset['val'] = flat_dataset['val'].add_column('captions', val_captions)
flat_dataset['test'] = flat_dataset['test'].add_column('captions', test_captions)

flat_dataset.save_to_disk('./datasets/flat_dataset_captions')