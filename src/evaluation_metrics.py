# Adopt from https://github.com/Alibaba-NLP/HiAGM/blob/master/train_modules/evaluation_metrics.py
import os 
import torch 


def _precision_recall_f1(right, predict, total):
    """
    Args:
        right: int, the count of right prediction
        predict: int, the count of prediction
        total: int, the count of labels

    Returns:
        p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, id2label, threshold=0.5, top_k=None, result_save_dir=None, save_flag='unknown'):
    """
    Agrs:
        epoch_labels: List[List[int]], ground truth, label id
        epoch_predicts: List[List[int]], predicted label_id
        vocab: data_modules.Vocab object
        threshold: Float, filter probability for tagging
        top_k: int, truncate the prediction
    
    Returns:
        confusion_matrix -> List[List[int]],
        Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        # Here we observe some classes will not appear in the test set and scores of these classes are set to 0.
        # If we exclude those classes, Macro-F1 will dramatically increase.
        # if gold_count_list[i] + predicted_count_list[i] != 0:
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Facilitate grouping and statistical analysis of experimental results
    if result_save_dir:
        with open(os.path.join(result_save_dir, f'Result_on_every_label{save_flag}.tsv'), 'w', encoding='utf8') as f:
            f.write('\t'.join(['Label', 'Precision', 'Recall', 'F1-score']) + '\n')
            for i, label in id2label.items():
                label = label + '_' + str(i)
                f.write('\t'.join([label, str(precision_dict[label]), str(recall_dict[label]), str(fscore_dict[label])]) + '\n')

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}


def get_path_set(slot2value, value2slot, id2label):
    
    all_labels = list(range(len(id2label))) 
    leaf_labels = list(set(all_labels).difference(set(list(slot2value.keys())))) 

    path_set = []

    for leaf_label in leaf_labels:
        cur = set()
        cur.add(leaf_label)
        parent = value2slot.get(leaf_label, -1)
        while parent != -1:
            cur.add(parent)
            parent = value2slot.get(parent, -1)
        path_set.append([cur, leaf_label])

    return path_set


def evaluate_based_on_path(epoch_predicts, epoch_labels, id2label, value2slot, slot2value, result_save_dir=None, save_flag='_unknown'):
    ''' Compute micro-F1 and macro-F1 based on the path unit. '''
    # path_set, value2slot = get_path_set(processor)
    path_set = get_path_set(slot2value, value2slot, id2label)
    epoch_gold = epoch_labels

    acc_right = 0
    acc_total = len(epoch_labels)

    predict_not_valid_atom_count = 0
    gold_atom_count = 0

    id2path = dict({i: path for i, path in enumerate(path_set)})  # All the label paths, but not necessarily the complete path from the root node to the leaf nodes.

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        # count for the gold and right items
        if len(set(sample_predict_id_list).intersection(set(sample_gold))) == len(sample_predict_id_list) \
                and len(sample_predict_id_list) == len(sample_gold):
            acc_right += 1
        predict_not_valid_atom_count += len(sample_predict_id_list)
        gold_atom_count += len(sample_gold)
    # evaluate acc based on the sample
    acc = acc_right / acc_total

    ## initialize confusion matrix
    # P-matrix
    right_count_list = [0 for _ in range(len(id2path))]
    gold_count_list = [0 for _ in range(len(id2path))]
    predicted_count_list = [0 for _ in range(len(id2path))]
    # C-matrix
    c_right_count_list = [0 for _ in range(len(id2label))]
    c_gold_count_list = [0 for _ in range(len(id2label))]
    c_predicted_count_list = [0 for _ in range(len(id2label))]

    wrong_atom_count = 0

    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        predict_path_idxs = []
        gold_path_idxs = []
        right_idxs = []
        # compute for our P-matrix
        for idx, path in enumerate(path_set):
            # count for predict confusion matrix
            path = path[0]
            if len(path.intersection(set(sample_predict_id_list))) == len(path):
                predicted_count_list[idx] += 1
                predict_path_idxs.append(idx)

            if len(path.intersection(set(sample_gold))) == len(path):
                gold_count_list[idx] += 1
                gold_path_idxs.append(idx)
        for right_idx in set(gold_path_idxs).intersection(predict_path_idxs):
            right_count_list[right_idx] += 1
            right_idxs.append(right_idx)
        valid_count = 0
        for idx in predict_path_idxs:
            valid_count += len(path_set[idx][0])
        wrong_atom_count += len(sample_predict_id_list) - valid_count 
        # compute for alibaba C-matrix
        # count for the gold and right items
    for sample_predict_id_list, sample_gold in zip(epoch_predicts, epoch_gold):
        gold_idxs = []
        right_idxs = []

        for gold in sample_gold:
            for label in sample_predict_id_list:
                if gold == label:
                    right_idxs.append(gold)
        for right_idx in right_idxs:
            flag = True
            parent = value2slot.get(right_idx, -1)

            while parent != -1:
                if parent not in right_idxs:
                    flag = False
                    break
                parent = value2slot.get(parent, -1)
            if flag:
                c_right_count_list[right_idx] += 1

        for gold in sample_gold:
            c_gold_count_list[gold] += 1
        # count for the predicted items
        for label in sample_predict_id_list:
            c_predicted_count_list[label] += 1

    ## P-matrix
    p_precision_dict = dict()
    p_recall_dict = dict()
    p_fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, path in id2path.items():
        leaf_label = path[1]
        label = str(leaf_label) + '_' + str(i)
        p_precision_dict[label], p_recall_dict[label], p_fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                                   predicted_count_list[i],
                                                                                                   gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    if result_save_dir:
        with open(os.path.join(result_save_dir, f'P-Metric_Result{save_flag}.tsv'), 'w', encoding='utf8') as f:
            f.write('\t'.join(['Label, P-Precision, P-Recall, P-F1-score']) + '\n')
            for i, path in id2path.items():
                leaf_label = path[1]
                label = str(leaf_label) + '_' + str(i)
                f.write('\t'.join([label, str(p_precision_dict[label]), str(p_recall_dict[label]), str(p_fscore_dict[label])]) + '\n')

    # PMacro-F1
    p_precision_macro = sum([v for _, v in p_precision_dict.items()]) / len(list(p_precision_dict.keys()))
    p_recall_macro = sum([v for _, v in p_recall_dict.items()]) / len(list(p_precision_dict.keys()))
    p_ori_macro_f1 = sum([v for _, v in p_fscore_dict.items()]) / len(list(p_fscore_dict.keys()))

    # PMicro-F1
    p_precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    p_recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    p_ori_micro_f1 = 2 * p_precision_micro * p_recall_micro / (p_precision_micro + p_recall_micro) \
        if (p_precision_micro + p_recall_micro) > 0 else 0.0
    x = wrong_atom_count / gold_atom_count
    a = 1 - 2 * (1 / (1 + torch.e ** (-x)) - 0.5)
    p_macro_f1 = a * p_ori_macro_f1
    p_micro_f1 = a * p_ori_micro_f1     

    ## C-matrix
    c_precision_dict = dict()
    c_recall_dict = dict()
    c_fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, leaf_label in id2label.items():
        label = leaf_label + '_' + str(i)
        c_precision_dict[label], c_recall_dict[label], c_fscore_dict[label] = _precision_recall_f1(
            c_right_count_list[i],
            c_predicted_count_list[i],
            c_gold_count_list[i]
        )
        right_total += c_right_count_list[i]
        gold_total += c_gold_count_list[i]
        predict_total += c_predicted_count_list[i]

    #Facilitate grouping and statistical analysis of experimental results
    if result_save_dir:
        with open(os.path.join(result_save_dir, f'C-Metric_Result{save_flag}.tsv'), 'w', encoding='utf8') as f:
            f.write('\t'.join(['Label, C-Precision, C-Recall, C-F1-score']) + '\n')
            for i, leaf_label in id2label.items():
                label = leaf_label + '_' + str(i)
                f.write('\t'.join([label, str(c_precision_dict[label]), str(c_recall_dict[label]), str(c_fscore_dict[label])]) + '\n')

    # CMacro-F1
    c_precision_macro = sum([v for _, v in c_precision_dict.items()]) / len(list(c_precision_dict.keys()))
    c_recall_macro = sum([v for _, v in c_recall_dict.items()]) / len(list(c_precision_dict.keys()))
    c_macro_f1 = sum([v for _, v in c_fscore_dict.items()]) / len(list(c_fscore_dict.keys()))
    # CMicro-F1
    c_precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    c_recall_micro = float(right_total) / gold_total
    c_micro_f1 = 2 * c_precision_micro * c_recall_micro / (c_precision_micro + c_recall_micro) \
        if (c_precision_micro + c_recall_micro) > 0 else 0.0
    result = {'p_precision': p_precision_micro,
              'p_recall': p_recall_micro,
              'p_micro_f1': p_micro_f1,
              'p_macro_f1': p_macro_f1,
              'p_ori_micro_f1': p_ori_micro_f1,
              'p_ori_macro_f1': p_ori_macro_f1,
              'c_precision': c_precision_micro,
              'c_recall': c_recall_micro,
              'c_micro_f1': c_micro_f1,
              'c_macro_f1': c_macro_f1,
              'P_acc': acc,
              'full': [p_precision_dict, p_recall_dict, p_fscore_dict, right_count_list, predicted_count_list,
                       gold_count_list,
                       c_precision_dict, c_recall_dict, c_fscore_dict, c_right_count_list, c_predicted_count_list,
                       c_gold_count_list,
                       ]
            }

    return result

