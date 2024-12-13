from datasets import Dataset, load_from_disk, DatasetDict
import math
import os

def dataset_with_captions(caption_filename):
    # Add captions column
    dataset = load_from_disk(os.path.join(os.path.dirname(__file__), 'flat_dataset'))
    
    with open(caption_filename, 'r') as f:
        captions = f.read().splitlines()
    
    dataset = dataset.add_column('caption', captions)
    
    return dataset


def get_splits(dataset):
    dataset = dataset.shuffle(seed=595)
    
    n_test = math.ceil(len(dataset) * 0.6)
    n_train = math.ceil(len(dataset) * 0.3)
    n_val = len(dataset) - (n_test + n_train)

    test_dataset = dataset.select(range(n_test)) # 60% test
    train_dataset = dataset.select(range(n_test, n_test+n_train)) # 30% train
    val_dataset = dataset.select(range(n_test+n_train, len(dataset))) # 10% val

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": val_dataset
    })
    
    return dataset

def dataset_captions_with_splits(caption_filename):
    return get_splits(dataset_with_captions(caption_filename))