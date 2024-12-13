from datasets import Dataset
import os, sys
import sklearn
import torch

sys.path.append('..')
sys.path.append(os.path.dirname(__file__))
from eval_sentence_sim import *
from generate_splits import dataset_captions_with_splits

with open(os.path.join(os.path.dirname(__file__), '..', 'categories.txt')) as f:
    categories = f.read().splitlines()


def gt_category_to_idx(gt):
    return torch.Tensor([categories.index(e) for e in gt]).long()


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


def evaluate_single(predictions, labels):
    '''
    Evaluates the results of a single model.
    
    Params:
        predictions (list[str]): The predicted actions
        labels (list[str]): the true actions
    '''
    
    compressed_captions = Dataset.from_dict(dict(predicted=predictions, label=labels))
    
    pred_idxs, pred_cats, pred_scores = get_predictions(compressed_captions, key='predicted')
    gt_idxs = gt_category_to_idx(labels)
    
    identical_acc = get_identical_acc(predictions, labels)
    print('Identical Accuracy:', identical_acc)

    topk_acc = dict()
    for topk in [1, 5, 20]:
        acc = get_topk_acc(pred_idxs, gt_idxs, topk=topk)
        topk_acc[topk] = acc.item()
        print(f'Top-{topk} acc: {acc}')
    
    pred_idxs_top1 = pred_idxs[:, 0]
    f1_score = sklearn.metrics.f1_score(gt_idxs, pred_idxs_top1, average='macro')
    
    # pred_scores_ordered = torch.zeros(len(predictions), len(categories))
    # pred_scores_ordered[torch.arange(len(predictions)).expand(-1, len(categories)), pred_idxs] = pred_scores
    # mAP = sklearn.metrics.average_precision_score(gt_idxs, pred_scores_top1, average='macro')
    
    return dict(
        identical_acc=identical_acc,
        topk_acc=topk_acc,
        f1_score=f1_score,
        # map=mAP
    )


def evaluate_from_file(filename, pred_key="predicted", label_key="label"):
    '''
    Evaluates from a saved file. The file should be pickled dictionary with keys "predicted" and "labels"
    
    Params:
        filename (str): The filename of the data file
        pred_key (str, optional): The key of the predictions from the dictionary
        label_key (str, optional): The key of the labels from the dictionary
    '''
    
    compressed_captions = torch.load(filename, weights_only=True)

    predictions = compressed_captions[pred_key]
    labels = compressed_captions[label_key]
    
    return evaluate_single(predictions, labels)


def evaluate_all():
    '''
    Evaluates all saved data files from a directory.
    
    Params:
        directory (str): The directory name where the data files reside
    '''
    
    caption_only_stats = dict()
    zsh_stats = dict()
    ft_stats = dict()
    
    # Evaluate captions
    print('===== Evaluating Captions Only =====')
    
    caption_directory = os.path.join(os.path.dirname(__file__), '../captioning/saved_captions/')
    for filename in os.listdir(caption_directory):
        full_name = os.path.join(caption_directory, filename)
        
        print('Evaluating Captions of ', filename)
        
        stats = evaluate_captions_from_file(full_name)
        name = filename.rstrip('.txt')
        caption_only_stats[name] = stats
        
        print()
    
    # Evaluate compressed captions
    print('===== Evaluating Compressed Captions =====')
    
    action_directory = os.path.join(os.path.dirname(__file__), '../compressed_captions/')
    for filename in os.listdir(action_directory):
        full_name = os.path.join(action_directory, filename)
        
        if not os.path.isdir(full_name):
            print('Evaluating Action of ', filename)
            stats = evaluate_from_file(full_name)
            
            if filename.endswith('_zsh.pt'):
                name = filename.rstrip('_zsh.pt')
                zsh_stats[name] = stats
            else:
                name = filename.rstrip('.pt')
                ft_stats[name] = stats
            
            print()
        
    return {
        'caption' : caption_only_stats,
        'zsh' : zsh_stats,
        'ft' : ft_stats
    }


def evaluate_captions_from_file(caption_filename):
    '''
    Evalutes the captions from the caption file
    '''
    
    dataset_captions = dataset_captions_with_splits(caption_filename)
    dataset_captions = dataset_captions['test']
    
    return evaluate_single(dataset_captions['caption'], dataset_captions['label'])


if __name__ == '__main__':
    evaluated = evaluate_all()
    print(evaluated)
    
    torch.save(evaluated, 'evaluations.pt')