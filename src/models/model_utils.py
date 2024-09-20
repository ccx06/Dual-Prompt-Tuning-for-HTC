import random 
import torch 


def sort_pred_labels_by_logits(depth2label, prediction_scores):
    """ Sort the output labels of each sample by probability.
    
    Args:
        prediction_scores: torch.Size(label_deep, label_num)
    """
    predict_labels = []

    end_pos = 0
    for layer, score in enumerate(prediction_scores):   
        predict_labels.append([])
        start_pos = end_pos
        end_pos = start_pos + len(depth2label[layer])
        layer_score = score[start_pos:end_pos]     
        sort_score, sort_idx = torch.sort(layer_score, descending=True)
        predict_labels[-1].extend([depth2label[layer][idx] for idx in sort_idx])    

    return predict_labels


def sort_pred_labels_by_verb_logits(depth2label, prediction_scores):
    """ Sort the output labels of each sample by probability.
    
    Args:
        prediction_scores: torch.Size(label_deep, label_num)
    """
    predict_labels = []
    for layer, nodes in depth2label.items():   
        layer_score = prediction_scores[nodes]  
        sort_score, sort_idx = torch.sort(layer_score, descending=True)
        predict_labels.append([depth2label[layer][idx] for idx in sort_idx])    

    return predict_labels


def negative_sampling(depth2label, pos_label_ids, sample_num_list, mode='random', output_logits=None):
    """ Sampling negative samples for each sample.

    Args:
        depth2label: 
        pos_label_ids: List[List[int]]，Positive labels at each label level.
        sample_num_list: List[int], Number of negative samples to be sampled each layer.
        mode: 
            'random': Completely random without maintaining hierarchical structure.
            'hard-hpt': Negative labels with the highest model score for `HPT` decoder.
            'hard-verb': Negative labels with the highest model score for `SoftVerbalizer` decoder.
        output_logits: torch. Probability matrix of model output. Required only when mode in {'hard-htp', 'hard-verb'}.
             if mode == 'hard-hpt': torch.Size([layer_depth, layer_num]).
             if mode == 'hard-verb': torch.Size([layer_num]).

    Returns:
        negatives: List[int], Sampled negative label IDs.
            
    """
    assert mode in {'random', 'hard-hpt', 'hard-verb'}, f"'{mode}' mode is not supported"
    negatives = []
    if mode == 'random':
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in label_list if c not in pos_label_ids[layer]]
            negatives.append(random.sample(candidates, k=min(len(candidates), sample_num_list[layer])))

    elif mode == 'hard-hpt':
        b_preds = sort_pred_labels_by_logits(depth2label, output_logits) 
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in b_preds[layer] if c not in pos_label_ids[layer]]
            negatives.append(candidates[:sample_num_list[layer]])

    elif mode == 'hard-verb':
        b_preds = sort_pred_labels_by_verb_logits(depth2label, output_logits) 
        for layer, label_list in depth2label.items():
            if len(pos_label_ids[layer]) == 0:
                candidates = []
            else:
                candidates = [c for c in b_preds[layer] if c not in pos_label_ids[layer]]
            negatives.append(candidates[:sample_num_list[layer]])

    return negatives


def compose_path(b_preds, slot2value, value2slot):
    """
    examples:
        b_preds: [[2, 103], [2, 68], [5, 131], [5, 85]]
    """
    b_path = []
    for pred in b_preds:
        path = []
        leaf_labels = list(set(pred).difference(set(list(slot2value.keys()))))

        for leaf in leaf_labels:
            cur_path = [leaf]
            parent = value2slot[leaf]
            while parent in pred:
                cur_path.append(parent)
                parent = value2slot[parent]

            path.append(cur_path)

        b_path.append(path)
    return b_path

     
def compute_rank_gt(pos_ids, children_ids, candidate_pairs):
    """ Get the true labels in ranking loss. 
    
    Args:
        pos_ids: List[int], The true label ids of the sample at the next level
        children_ids: List[int], The children nodes of the current layer label.
        candidate_pairs: List[tuple(int)], List of ids that need to be sorted.

    Returns:
        gt_list: List[int]
    """
    gt = []
    for a, b in candidate_pairs:
        # Positive child-labels > Negative child-labels > Non child-labels
        a_flag = 2 if a in pos_ids else 1 if a in children_ids else 0
        b_flag = 2 if b in pos_ids else 1 if b in children_ids else 0
        if a_flag > b_flag:
            gt.append(1)
        elif a_flag == b_flag:
            gt.append(0)
        else:
            gt.append(-1)

    return gt


def is_all_ones(arr):
    for num in arr:
        if num != 1:
            return 0
    return 1

def judge_consistency_path(b_preds, true_path_set):
    """ Judging whether the model prediction can form a consistency path. """
    res = []
    for pred in b_preds:
        r = []
        for p in pred:
            if set(p) in true_path_set:
                r.append(1)
            else:
                r.append(0)
        res.append(r)
    return res 


def judge_correct_pred(b_preds, b_truth):
    """ Judging whether the model prediction belongs to real label. """
    res = []
    for pred in b_preds:
        r = []
        if set(pred) in b_truth:
            r.append(1)
        else:
            r.append(0)
        res.append(r)
    return res 


def get_all_path_set(depth2label, value2slot):
    """ Retrieve all existing paths in the tag tree, including incomplete paths that do not require running from root to leaf node (This is the difference from the `get_path_set` function in the upper level directory of utils.py). """
    path_set = []

    depth = len(depth2label)

    for i in range(1, depth):
        for node in depth2label[i]:
            cur = set()
            cur.add(node)
            parent = value2slot.get(node, -1)
            while parent != -1:
                cur.add(parent)
                parent = value2slot.get(parent, -1)
            path_set.append(cur)
    return path_set
            