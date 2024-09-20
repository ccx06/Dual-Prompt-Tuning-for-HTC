"""  
@Desc: Construct dataset: train.json, dev.json, test.json, slot.pt, .taxonomy, value_dict.pt, label_symbol_dict.pt
"""
import json 
import torch
from collections import defaultdict


def build_label_dict(label_vocab_file):
    label_dict = {}
    with open(label_vocab_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            label_dict[i] = line.strip()
    return label_dict 


def build_dataset(data_file, save_file, label2id):
    with open(data_file, 'r', encoding='utf8') as fr, \
        open(save_file, 'w', encoding='utf8') as fw:
        for line in fr:
            sample = json.loads(line.strip())
            fw.write(json.dumps({
                'token': sample['token'], 
                'label': [label2id[l] for l in sample['label']]
            }, ensure_ascii=False) + '\n')
    print('Build dataset: ', save_file)
    

def build_slot2value(hier_file, label2id):
    slot2value = defaultdict(list)
    with open(hier_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) > 1 and line[0] in label2id and line[1] in label2id:
                slot2value[label2id[line[0]]].append(label2id[line[1]])
    return slot2value 
     

if __name__ == "__main__":
    label_vocab_file = 'label.vocab'
    label_dict = build_label_dict(label_vocab_file)
    torch.save(label_dict, 'value_dict_labelname.pt')
    
    hier_file = 'hierarchy.txt'
    label2id = {v: k for k, v in label_dict.items()}
    slot2value = build_slot2value(hier_file, label2id)
    torch.save(slot2value, 'slot.pt')
    
    data_file = 'pre_bgc_train.json'
    save_file = 'bgc_train.json'
    build_dataset(data_file, save_file, label2id)
    
    data_file = 'pre_bgc_dev.json'
    save_file = 'bgc_dev.json'
    build_dataset(data_file, save_file, label2id)
    
    data_file = 'pre_bgc_test.json'
    save_file = 'bgc_test.json'
    build_dataset(data_file, save_file, label2id)
    
    label_symbol_mapper_file = 'label_symbol_mapper.txt'
    new_token_file = 'special_tokens.txt'
    new_value_dict = {}
    label_symbol_mapper = {}
    i = 0
    for k, v in label_dict.items():
        new_value_dict[k] = f'L{i}'
        label_symbol_mapper[v] = f'L{i}'
        i += 1

    torch.save(new_value_dict, 'value_dict.pt')
    torch.save(label_symbol_mapper, 'label_symbol_dict.pt')

    with open(label_symbol_mapper_file, 'w', encoding='utf8') as f:
        for k, v in label_symbol_mapper.items():
            f.write('\t'.join([k, v]) + '\n')
            
    with open(new_token_file, 'w', encoding='utf8') as f:
        f.write('L-L' + '\n')
        for v in label_symbol_mapper.values():
            f.write(v + '\n')

    print("Done!")

