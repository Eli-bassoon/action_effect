from datasets import Dataset, DatasetDict
from load_dataset import dataset
import math

train_split = 0.2

train = {
    'images': [],
    'action': []
}
test = {
    'images': [],
    'action': []
}

for category in dataset['ActionEffect']:
    n = len(category['positive_image_list'])
    train_n = math.ceil(n*train_split)
    test_n = n - train_n
    
    train['images'].extend(category['positive_image_list'][:train_n])
    train['action'].extend([category['verb noun']] * train_n)
    
    test['images'].extend(category['positive_image_list'][train_n:])
    test['action'].extend([category['verb noun']] * test_n)

flat_dataset = DatasetDict({
    'train' : Dataset.from_dict(train),
    'test' : Dataset.from_dict(test)
})
flat_dataset.save_to_disk('flat_dataset')