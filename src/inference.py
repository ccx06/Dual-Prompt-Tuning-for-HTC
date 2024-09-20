"""
Desc: Model Infer.
"""

import torch
import os
from transformers import AutoTokenizer
from models.dual_prompt_model import HTCDualPromptModel


class ModelInference(object):
    """
    Load Model and Infer labels.
    """

    def __init__(self, arch, model_path, config_dir):
        # model
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(arch)

        # label info 
        self.config_dir = config_dir 
        self.label_dict = None 
        self.labelid2name = None
        self.path_list = None 
        self.depth2label=None 
        self.num_class = None 
        self.max_depth = None 
        self.prefix = None 
        self._init_label_settings()
        
        # model 
        checkpoint = torch.load(model_path, map_location='cpu')

        self.model = HTCDualPromptModel.from_pretrained(arch,
                                    tokenizer=self.tokenizer,
                                    depth2label=self.depth2label,
                                    label_dict=self.label_dict,)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint['param'])
        else:
            self.model.load_state_dict(checkpoint['param'])
        self.model.to('cuda')
        self.model.eval()


    def _init_label_settings(self):
        label_symbol_mapper = torch.load(os.path.join(self.config_dir, 'label_symbol_dict.pt'))  # key: label name, value: label symbol
        symbol2label_name = {v:k for k, v in label_symbol_mapper.items()}  
        self.label_dict = torch.load(os.path.join(self.config_dir, 'value_dict.pt'))  # key: label id, value: label symbol
        self.labelid2name = {k: symbol2label_name[v] for k, v in self.label_dict.items()}
        
        slot2value = torch.load(os.path.join(self.config_dir, 'slot.pt'))   
        value2slot = {}  
        num_class = 0
        for s in slot2value:
            for v in slot2value[s]:
                value2slot[v] = s
                if num_class < v:
                    num_class = v
        num_class += 1
        self.path_list = [(i, v) for v, i in value2slot.items()] 
        for i in range(num_class):
            if i not in value2slot:
                value2slot[i] = -1

        def get_depth(x):
            depth = 0
            while value2slot[x] != -1:
                depth += 1
                x = value2slot[x]
            return depth

        depth_dict = {i: get_depth(i) for i in range(num_class)}
        self.max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
        self.depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(self.max_depth)}    # dict， key表示第key层, value为在第key层的标签id， 如{0: [0, 1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10, 11, 12, 13, 14, 15, ...]}

        for depth in self.depth2label:
            for l in self.depth2label[depth]:
                self.path_list.append((num_class + depth, l))

        self.num_class = num_class
        template = 'it belongs to'
        prefix = [self.tokenizer.convert_tokens_to_ids(t) for t in template.split(' ')]

        for i in range(self.max_depth):
            prefix.append(self.tokenizer.mask_token_id)
            if i < self.max_depth-1:
                prefix.append(self.tokenizer.convert_tokens_to_ids('L-L'))     
                # prefix.append(tokenizer.convert_tokens_to_ids('-'))

        template = 'rather than'
        prefix.extend([self.tokenizer.convert_tokens_to_ids(t) for t in template.split(' ')])
        for i in range(self.max_depth):
            prefix.append(self.tokenizer.mask_token_id)
            if i < self.max_depth-1:
                prefix.append(self.tokenizer.convert_tokens_to_ids('L-L'))
                # prefix.append(tokenizer.convert_tokens_to_ids('-'))

        prefix.append(self.tokenizer.sep_token_id)
        self.prefix = prefix

    def tokenize(self, batch):

        input_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for t in batch:
            tokens = self.tokenizer(t, truncation=True)
            input_ids = tokens['input_ids'][:-1][:512 - len(self.prefix)] + self.prefix
            input_batch['input_ids'].append(input_ids)
            input_batch['input_ids'][-1].extend([self.tokenizer.pad_token_id] * (512 - len(input_batch['input_ids'][-1])))
            input_batch['attention_mask'].append(tokens['attention_mask'][:-1][:512 - len(self.prefix)] + [1] * len(self.prefix))
            input_batch['attention_mask'][-1].extend([0] * (512 - len(input_batch['attention_mask'][-1])))
            input_batch['token_type_ids'].append([0] * 512)
        
        input_batch['input_ids'] = torch.Tensor(input_batch['input_ids']).long()
        input_batch['attention_mask'] = torch.Tensor(input_batch['attention_mask']).long()
        input_batch['token_type_ids'] = torch.Tensor(input_batch['token_type_ids']).long()

        return input_batch


    def predict(self, sentences):
        batch = self.tokenize(sentences)
        pred_labels = []
        with torch.no_grad():
            inputs = {k: v.to('cuda') for k, v in batch.items()}
            output_ids, logits = self.model.generate(inputs['input_ids'], depth2label=self.depth2label,)
            for out, logit in zip(output_ids, logits):
                pred_labels.append([self.labelid2name[o] for o in out])
        return pred_labels


if __name__ == "__main__":
    
    model = ModelInference(arch='/mnt/home/pretrained_models/bert-base-uncased-bgc',
                        model_path='checkpoints/bgc_lambda1-0.5_lambda2-0.2_alpha-0.6_beta-0.1_2024091810/checkpoint_best_micro.pt',
                        config_dir='data/BGC')

    sentences = ["Teenage Mutant Ninja Turtles: The Box Set Volume 1: TMNT co-creator Kevin Eastman and writer Tom Waltz guide readers through a ground-breaking new origin and into epic tales of courage, loyalty and family as the Turtles and their allies battle for survival against enemies old and new in the dangerous streets and sewers of New York City. Includes TMNT volumes #1–5, which collects the first 20 issues of the ongoing series.",
    "Betty & Veronica: Fairy Tales: Take a magical trip down Storybook Lane with this collection of the best fairy tale stories ever told in Riverdale! From glass slippers to giant beanstalks and yellow brick roads to tales from under the seas, this classic collection will have you entertained for hours!",]
    labels = model.predict(sentences)
    print(labels)

    