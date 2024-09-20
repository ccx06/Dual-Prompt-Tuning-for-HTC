import torch
import random
import numpy as np
import logging


def seed_torch(seed=1029):
    print('Set seed to', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def constraint(batch_id, input_ids, label_dict):
    last_token = input_ids[-1].item()
    if last_token not in label_dict:
        ret = [2]
    else:
        ret = [i + 3 for i in label_dict[input_ids[-1].item() - 3]]
    return ret


def init_logger(log_path):
    logging.basicConfig(filemode='w')
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8', mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s -   %(message)s', 
                                  datefmt='%m/%d/%Y %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


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
    