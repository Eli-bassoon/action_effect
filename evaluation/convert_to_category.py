from sentence_transformers import SentenceTransformer
from load_dataset import dataset
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

# The categories
categories = [
    row['verb noun'] for row in dataset['ActionEffect']
]

# Calculate embeddings for categories
categories_vec = model.encode(categories)

def convert_to_category(predictions, topk=1):
    predictions_vec = model.encode(predictions)

    # Calculate the embedding similarities
    similarities = model.similarity(predictions_vec, categories_vec)
    
    if topk == -1:
        vals, idxs = torch.sort(similarities, descending=True)
    else:
        vals, idxs = similarities.topk(topk)
        
    category_names = [[categories[i] for i in this_idxs] for this_idxs in idxs] 
    
    return idxs, category_names, vals