from sentence_transformers import SentenceTransformer
import torch
import os

# The categories
with open(os.path.join(os.path.dirname(__file__), '..', 'categories.txt')) as f:
    categories = f.read().splitlines()

model = SentenceTransformer("all-MiniLM-L6-v2")

BG_THRESH = 0.4
BG_IDX = len(categories)

# Calculate embeddings for categories
categories_vec = model.encode(categories[:-1])

def get_similarities(predictions):
    predictions_vec = model.encode(predictions)

    # Calculate the embedding similarities
    similarities = model.similarity(predictions_vec, categories_vec)
    
    return similarities


def convert_to_category(predictions, topk=1):
    similarities = get_similarities(predictions)
    
    vals, idxs = torch.sort(similarities, descending=True)
    category_names = [[categories[i] for i in this_idxs] for this_idxs in idxs]
    
    # # Expand to include background class
    # vals_exp = torch.zeros((vals.shape[0], vals.shape[1]+1))
    # idxs_exp = torch.zeros((idxs.shape[0], idxs.shape[1]+1))
    
    # # Add background category
    # for i in range(len(predictions)):
    #     if vals[i, 0] < BG_THRESH:
    #         vals_exp[i, 0] = 1
    #         vals_exp[i, 1:] = vals[i, :]
            
    #         idxs_exp[i, 0] = BG_IDX
    #         idxs_exp[i, 1:] = idxs[i, :]
            
    #         category_names[i].insert(0, 'background')
    #     else:
    #         vals_exp[i, -1] = -1
    #         vals_exp[i, :-1] = vals[i, :]
            
    #         idxs_exp[i, -1] = BG_IDX
    #         idxs_exp[i, :-1] = idxs[i, :]
            
    #         category_names[i].append('background')
    
    # vals = vals_exp
    # idxs = idxs_exp.long()
    
    if topk != -1:
        vals, idxs = vals[:, :topk], idxs[:, :topk]
    
    return idxs, category_names, vals


def get_predictions(dataset, key='caption'):
    '''
    Gets the predicted category distribution. For each prediction in a dataset, finds the most similar actions of the possible categories
    '''
    
    # Get predictions
    pred_idxs = []
    pred_cats = []
    pred_scores = []
    
    for batch in dataset.iter(batch_size=200):
        batch_idxs, batch_cats, batch_scores = convert_to_category(batch[key], topk=-1)
        pred_idxs.append(batch_idxs)
        pred_cats.extend(batch_cats)
        pred_scores.append(batch_scores)
    
    pred_idxs = torch.cat(pred_idxs, dim=0).long()
    pred_scores = torch.cat(pred_scores, dim=0)

    pred_cats_temp = []
    for cat in pred_cats:
        pred_cats_temp.extend(cat)

    pred_cats = pred_cats_temp
    
    return pred_idxs, pred_cats, pred_scores
