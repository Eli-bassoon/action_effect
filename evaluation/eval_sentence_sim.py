from convert_to_category import convert_to_category
import torch

def get_predictions(dataset, key='caption'):
    '''
    Gets the predicted category distribution
    '''
    # Get predictions
    pred_idxs = []
    pred_cats = []
    
    for batch in dataset.iter(batch_size=200):
        batch_idxs, batch_cats, _ = convert_to_category(batch[key], topk=-1)
        pred_idxs.append(batch_idxs)
        pred_cats.extend(batch_cats)
    
    pred_idxs = torch.cat(pred_idxs, dim=0)

    pred_cats_temp = []
    for cat in pred_cats:
        pred_cats_temp.extend(cat)

    pred_cats = pred_cats_temp
    
    return pred_idxs, pred_cats


def get_identical_acc(answers, gt_cats):
    '''
    Gets the identical accuracy where each character must match.
    '''
    correct = 0
    total = 0
    
    for i in range(len(answers)):
        if answers[i] == gt_cats[i]:
            correct += 1
        
        total += 1
    
    return correct / total
    

def get_topk_acc(pred, gt, topk=1):
    '''
    Gets the top-k accuracy from the predicted categories.
    '''
    pred = pred[:, :topk]
    
    in_topk = torch.any(pred == gt.view(-1, 1), dim=1)
    
    return in_topk.sum() / pred.shape[0]
