"""
@Desc: Extracting structured samples from decompressed data.
"""
import re 
import json 


def extract_tile_body(file_path, save_path, label_vocab_save_path=None):
    num = 0
    labels_set = set()
    with open(file_path, 'r', encoding='utf8') as fr, \
        open(save_path, 'w', encoding='utf8') as fw:
            
        meta_title, meta_body, meta_labels = '', '', []
        for line in fr:
            line = line.strip()
            if line.startswith('<title>'):
                meta_title = re.findall(r'<title>(.*)</title>', line)[0]
            
            if line.startswith('<body>'):
                meta_body = re.findall(r'<body>(.*)</body>', line)[0]
            
            if line.startswith('<d'):
                label_level0 = re.findall(r'<d0>(.*?)</d0>', line)
                label_level1 = re.findall(r'<d1>(.*?)</d1>', line)
                label_level2 = re.findall(r'<d2>(.*?)</d2>', line)             
                label_level3 = re.findall(r'<d3>(.*?)</d3>', line)
                if len(label_level0) > 0:
                    meta_labels.extend(label_level0)
                if len(label_level1) > 0:
                    meta_labels.extend(label_level1)
                if len(label_level2) > 0:
                    meta_labels.extend(label_level2)
                if len(label_level3) > 0:
                    meta_labels.extend(label_level3)
            
            if len(meta_title) > 0 and len(meta_body) > 0 and len(meta_labels) > 0:
                num += 1
                fw.write(json.dumps({'token': ': '.join([meta_title, meta_body]), 'label': meta_labels}, ensure_ascii=False) + '\n')
                for la in meta_labels:
                    labels_set.add(la)
                meta_title, meta_body, meta_labels = '', '', []
    
    if label_vocab_save_path:
        with open(label_vocab_save_path, 'w', encoding='utf8') as f:
            for label in labels_set:
                f.write(label + '\n')
                
    print('Number of labels: ', len(labels_set))
    print('Number of samples: ', num)
    print('Finish writing: ', save_path)
    

def build_label_vocab(label_file, vocab_save_path):
    label_set = set()
    with open(label_file, 'r', encoding='utf8') as fr, \
        open(vocab_save_path, 'w', encoding='utf8') as fw:
            for line in fr:
                labels = line.strip().split('\t')
                if len(labels) > 2:
                    print()
                for la in labels:
                    la = la.strip()
                    if la not in label_set:   
                        fw.write(la + '\n')
                        label_set.add(la)
    print(len(label_set))
    print('Finish writing: ', vocab_save_path)
        
        
if __name__ == "__main__":
    file_path = 'BlurbGenreCollection_EN_train.txt'
    save_path = 'pre_bgc_train.json'
    vocab_save_path = 'label.vocab'
    extract_tile_body(file_path, save_path, label_vocab_save_path=vocab_save_path)
    
    file_path = 'BlurbGenreCollection_EN_test.txt'
    save_path = 'pre_bgc_test.json'
    extract_tile_body(file_path, save_path)
    
    file_path = 'BlurbGenreCollection_EN_dev.txt'
    save_path = 'pre_bgc_dev.json'
    extract_tile_body(file_path, save_path)
    
    # label_file = 'hierarchy.txt'
    # build_label_vocab(label_file, vocab_save_path)           