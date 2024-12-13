from datasets import load_dataset, Dataset
from PIL import Image

dataset = load_dataset("sled-umich/Action-Effect", trust_remote_code=True)

images = []
labels = []

for category in dataset['ActionEffect']:
    print(category['verb noun'])
    
    images.extend(category['positive_image_list'])
    labels.extend([category['verb noun']] * len(category['positive_image_list']))

flat_dataset = Dataset.from_dict(dict(image=images, label=labels))
flat_dataset.save_to_disk('./flat_dataset')