"""  
@Desc: Construct dataset: train.json, dev.json, test.json, slot.pt, .taxonomy, value_dict.pt, label_symbol_dict.pt
"""
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict
import datasets

np.random.seed(7)


def get_label_mapper():
    label_symbol_mapper = {}
    label_tree_file = 'wos.taxnomy.ori'
    new_label_tree_file = 'wos.taxnomy'

    with open(new_label_tree_file, 'w', encoding='utf8') as fw, \
         open(label_tree_file, 'r', encoding='utf8') as fr:
        for i, line in enumerate(fr):
            new_line = []
            if i == 0:
                level1 = line.strip().split('\t')
                new_line.append('Root')
                for k, l1 in enumerate(level1):
                    if l1 != 'Root':
                        label_symbol_mapper[l1] = f'L{k-1}'
                        new_line.append(label_symbol_mapper[l1])
                fw.write('\t'.join(new_line) + '\n')
                        
            else:
                level2 = line.strip().split('\t')
                level1_name = level2[0]
                new_line.append(label_symbol_mapper[level1_name])
                for j, l2 in enumerate(level2[1:]):
                    label_symbol_mapper[l2] = f'{label_symbol_mapper[level1_name]}-{j}'
                    new_line.append(label_symbol_mapper[l2])
                fw.write('\t'.join(new_line) + '\n')
    
    return label_symbol_mapper


if __name__ == '__main__':
    label_symbol_mapper = get_label_mapper()
    special_token_save_file = 'special_tokens.txt'
    label_mapper_save_file = 'label_symbol_mapper.txt'

    with open(special_token_save_file, 'w', encoding='utf8') as f:
        f.write('L-L' + '\n') 
        for v in label_symbol_mapper.values():
            f.write(v + '\n')

    with open(label_mapper_save_file, 'w', encoding='utf8') as f:
        for k, v in label_symbol_mapper.items():  
            f.write('\t'.join([k, v]) + '\n')

    torch.save(label_symbol_mapper, 'label_symbol_dict.pt')

    # tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    label_dict = {}
    hiera = defaultdict(set)
    data = datasets.load_dataset('json', data_files='wos_total.json')['train']
    for l in data['doc_label']:
        if l[0] not in label_dict:
            label_dict[l[0]] = len(label_dict)
    for l in data['doc_label']:
        assert len(l) == 2
        if l[1] not in label_dict:
            label_dict[l[1]] = len(label_dict)
        hiera[label_dict[l[0]]].add(label_dict[l[1]])
    value_dict = {i: label_symbol_mapper[v] for v, i in label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(hiera, 'slot.pt')

    id = [i for i in range(len(data))]
    np_data = np.array(id)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()
    with open('WebOfScience_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
