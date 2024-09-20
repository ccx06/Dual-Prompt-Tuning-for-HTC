"""
@Desc: Model architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import itertools
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead, BertForMaskedLM, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from openprompt.prompts import SoftVerbalizer

from .loss import multilabel_categorical_crossentropy
from .model_utils import *


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class VerbBertForMaskedLM(BertForMaskedLM):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, bert, cls):
        super().__init__(config)

        self.bert = bert
        self.cls = cls
        # # Initialize weights and apply final processing
        # self.post_init()


class HTCDualPromptModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, 
                layer=1, 
                mlmloss_weight=1, 
                multitask_loss_weight=0.5, 
                cont_configs=None, 
                logger=None,
                depth2label=None, 
                label_dict=None,
                value2slot=None,
                slot2value=None,
                **kwargs):
        super().__init__(config)

        self.model_config = config
        self.bert = BertModel(config, add_pooling_layer=False) 
        self.tokenizer = tokenizer
        self.cls = BertOnlyMLMHead(config)
        self.multi_task_mlp = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            NormedLinear(self.model_config.hidden_size, 2)
        )

        self.vocab_size = self.tokenizer.vocab_size
        self.depth2label = depth2label
        self.value2slot = value2slot
        self.slot2value = slot2value
        self.label_dict = label_dict
        self.layer = layer
        self.logger = logger
        self.num_labels = len(label_dict)

        self.path_set = None
        self.label_inputs_dict = None
        if depth2label and value2slot:
            self.path_set = get_all_path_set(depth2label, value2slot)   # All legitimate label path collections (not required to be complete, i.e. from root to leaf label)
            self.label_inputs_dict = self.convert_single_label_to_hierarchical_label_seq()

        self.cont_configs = cont_configs
        self.mlmloss_weight = mlmloss_weight
        self.multitask_loss_weight = multitask_loss_weight

        verbalizer_list = self.build_verbalzier() 
        self.verbLength = len(verbalizer_list)
        for idx, verbalizer in enumerate(verbalizer_list):
            self.__setattr__(f"verbalizer{idx}", verbalizer)
        self.init_weights()


        self.rank_loss_func = nn.MarginRankingLoss(margin=0.0)


    def build_verbalzier(self):
        verbalizer_list = []
        for layer, node in self.depth2label.items():
            label_list = [self.label_dict[n] for n in node]
            verbalizer_list.append(SoftVerbalizer(self.tokenizer, model=VerbBertForMaskedLM(self.model_config, self.bert, self.cls), classes=label_list))
        
        return verbalizer_list
        

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_verbalizer_embedding(self):
        self.logger.info("using label embeddings for soft verbalizer initialization.")
        depth = len(self.depth2label)
        label_emb_list = []
        for idx in range(depth):
            nodes = self.depth2label[idx]
            nodes.sort()
            label_dict = [self.label_dict[i] for i in nodes]
            label_dict = dict({i:v for i, v in enumerate(label_dict)})
            label_dict = {i: self.tokenizer.encode(v) for i, v in label_dict.items()}
            label_emb = []
            input_embeds = self.get_input_embeddings()   # Embedding(vocab_size, hidden_size, padding_idx=0)

            for i in range(len(label_dict)):
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
            label_emb = torch.stack(label_emb)   # torch.Size([label_num, hidden_size])
            label_emb_list.append(label_emb)

        for depth_idx in range(depth - 2, -1, -1):
            layer_label_emb = label_emb_list[depth_idx]
            layer_labels = self.depth2label[depth_idx]
            for i, label in enumerate(layer_labels):
                if label not in self.slot2value:
                    continue
                idx = [self.depth2label[depth_idx+1].index(x) for x in list(self.slot2value[label])]
                layer_label_emb[i] = layer_label_emb[i] + label_emb_list[depth_idx + 1][idx, :].mean(dim=0)
            label_emb_list[depth_idx] = layer_label_emb
 

        for idx in range(depth):
            label_emb = label_emb_list[idx]
            getattr(self.__getattr__(f"verbalizer{idx}").head.predictions,
                    'decoder').weight.data = label_emb
            getattr(self.__getattr__(f"verbalizer{idx}").head.predictions,
                    'decoder').weight.data.requires_grad = True


    def convert_single_label_to_hierarchical_label_seq(self):
        """Convert single-label into label sequence with hierarchical infomation. For example:
        [CLS] {label name} [SEP] [{parent node}][SEP]{child nodes} [SEP]
        Note that [CLS] and [SEP] will not be added in the current step.

        Args:
            label_name: List[str].
        
        Returns:
            hier_label_seq: Dict[(str, str)], Each element corresponds to a label(key) corresponding to a sequence with hierarchical info(value).
        """
        hier_label_seq = OrderedDict()   # Maintain the same order as slot_label_list

        join_symbol = ' ' + self.tokenizer.sep_token + ' '
        for level, label_ids in self.depth2label.items():
            if level == 0:          # first layer 
                for idx in label_ids:
                    label = self.label_dict[idx]
                    if idx in self.slot2value:
                        children = ','.join([self.label_dict[i] for i in self.slot2value[idx]])
                    else:
                        # if it's leaf node 
                        children = 'none'
                    seq = [label, 'root', children]
                    seq = join_symbol.join(seq)
                    hier_label_seq[idx] = seq 

            
            elif level == len(self.depth2label) - 1:   # last layer, leaf nodes 
                for idx in label_ids:
                    label = self.label_dict[idx]
                    parent = self.label_dict[self.value2slot[idx]]
                    seq = join_symbol.join([label, parent, 'none'])
                    hier_label_seq[idx] = seq 
            
            else:
                for idx in label_ids:
                    label = self.label_dict[idx]
                    if idx in self.slot2value:
                        children = ','.join([self.label_dict[i] for i in self.slot2value[idx]])
                    else:
                        # if it's leaf node 
                        children = 'none'
                    seq = [label, 'root', children]
                    parent =self.label_dict[self.value2slot[idx]]
                    seq = join_symbol.join([label, parent, children])
                    hier_label_seq[idx] = seq 
            
        return hier_label_seq


    def get_label_embeddings(self, label_ids):
        """ Obtain label embeddings (as prototype for each label) by PLMs. """
        label_ids = list(set(label_ids))
        batch_indexed_tokens = []
        batch_attention_mask = []
        batch_segment_ids = []

        max_seq_len = 0
        for label_id in label_ids:
            text = self.label_inputs_dict[label_id]
            # add special tokens
            tokenized_text = [self.tokenizer.cls_token] + self.tokenizer.tokenize(text) + [self.tokenizer.sep_token]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            batch_indexed_tokens.append(indexed_tokens)

            attention_mask = [1] * len(indexed_tokens)
            batch_attention_mask.append(attention_mask)

            if len(indexed_tokens) > max_seq_len:
                max_seq_len = len(indexed_tokens)
        # pad
        for i, indexed_tokens in enumerate(batch_indexed_tokens):
            batch_segment_ids.append([0] * max_seq_len)

            if len(indexed_tokens) < max_seq_len:
                padding_length = max_seq_len - len(indexed_tokens)
                batch_indexed_tokens[i] = indexed_tokens + ([self.tokenizer.pad_token_id] * padding_length)
                batch_attention_mask[i] = batch_attention_mask[i] + ([self.tokenizer.pad_token_id] * padding_length)
                
        all_input_ids = torch.tensor(batch_indexed_tokens, dtype=torch.long).to(self.device)
        all_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)
        all_token_type_ids = torch.tensor(batch_segment_ids, dtype=torch.long).to(self.device)

        # Embed
        label_embeddings = {}
        # for b, (input_ids, attention_mask, token_type_ids) in enumerate(batch_loader(16, all_input_ids, all_attention_mask, all_token_type_ids)):
        pooled_output = self.bert(all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_token_type_ids)[0]
        for i in range(pooled_output.shape[0]):
            out = pooled_output[i]
            label_name = self.label_dict[label_ids[i]]   
            label_name_length = len(self.tokenizer.tokenize(label_name))
            label_name_emb = torch.mean(out[1:1+label_name_length], dim=0)
            
            label_embeddings[label_ids[i]] = label_name_emb

        return label_embeddings


    def calculate_contrastive_loss(self, pos_labels, neg_labels, neg_outputs_at_mask, label_token_repr, layer_atten=None):
        """  
        Hierarchy-aware Peer-label Contrastive Leraning module.
        Args:
            pos_labels: List[List[int]], shape=[bz, label_deep, sample_num]. Each element is a list of positive sample label IDs for each sample.            
            neg_labels: List[List[int]], shape=[bz, label_deep, sample_num. Each element is a list of negative sample label IDs for each sample.
            neg_outputs_at_mask: torch.Size(bz, label_depth, dimension). The embeddings on the negative mask positions.
            label_token_repr: torch.tensor, torch.Size(bz, label_depth, dim). The representation vector of the positive label.
            layer_atten: torch.tensor, torch.Size(label_depth). Contrastive Loss weights on each layer.

        Return:
            contrastive_loss: float. 
        
        """
        contrastive_loss = 0 
        rank_loss = 0 

        bz = len(pos_labels)
        label_depth = len(self.depth2label)

        participate_labels = list(set([n for neg in neg_labels for ne in neg for n in ne])) + list(set([p for pos in pos_labels for po in pos for p in po]))
        labels_emb_dict = self.get_label_embeddings(label_ids=participate_labels)

        all_pos_emb, all_neg_emb = [[] for _ in range(bz)], [[] for _ in range(bz)]
        
        # Record all layers of all samples.
        all_layers_neg_emb = [[[] for _ in range(label_depth)] for _ in range(bz)]
        all_layers_pos_emb = [[[] for _ in range(label_depth)] for _ in range(bz)]
        all_layers_pos = [[[] for _ in range(label_depth)] for _ in range(bz)]
        all_layers_neg = [[[] for _ in range(label_depth)] for _ in range(bz)]

        # Compute on each layer.
        for layer, labels in self.depth2label.items(): 
            # For each layer
            ## positives -- Labels for each sample
            for i, b_la in enumerate(pos_labels):    # --- batch
                for level_la in b_la:      # -- each layer
                    for la in level_la:    # -- label list of each sample -- Compatible with multi label classification
                        if la in labels:      
                            all_layers_pos[i][layer].append(la)
                            all_pos_emb[i].append(labels_emb_dict[la]) 
                            all_layers_pos_emb[i][layer].append(labels_emb_dict[la])

            ## negatives 
            for i, b_la in enumerate(neg_labels):
                for level_la in b_la:
                    for la in level_la:
                        if la in labels:      
                            all_layers_neg[i][layer].append(la)
                            all_neg_emb[i].append(labels_emb_dict[la])
                            all_layers_neg_emb[i][layer].append(labels_emb_dict[la])
            
            ## Contrastive learning
            each_layer_cont_loss = 0
            for i in range(bz):
                if len(all_pos_emb[i]) == 0 or len(all_neg_emb[i]) == 0:
                    continue  
                
                # 1. Positive Label Contrastive Learning. 
                # Objects: positive label mask tokens.
                # Positives：M ground truth labels.
                # Negatives: The sampled K negative labels and negative label mask token
                query_vector = nn.functional.normalize(label_token_repr[i,layer,:], dim=0)    # torch.Size(dim)   
                pos_key_matrix = nn.functional.normalize(torch.stack(all_pos_emb[i], dim=0), dim=1)    # torch.Size(pos_num, dim)
                neg_key_matrix = nn.functional.normalize( \
                    torch.cat([torch.stack(all_neg_emb[i], dim=0), neg_outputs_at_mask[i, layer, :].unsqueeze(dim=0)], dim=0), dim=1)

                pos_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), pos_key_matrix)
                neg_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), neg_key_matrix)

                numerator = torch.sum(torch.exp(pos_sim / self.cont_configs.tau))
                denominator = numerator + torch.sum(torch.exp(neg_sim / self.cont_configs.tau))
                pos_dual_cont_loss = -torch.log(numerator / denominator)

                # 2. Negative Label Contrastive Learning. 
                # Objects: negative label mask tokens.
                # Positives: negative labels of this instance.
                # Negatives: ground-truth labels and positive label mask token.
                query_vector = nn.functional.normalize(neg_outputs_at_mask[i, layer, :], dim=0)  
                pos_key_matrix = nn.functional.normalize(torch.stack(all_neg_emb[i], dim=0), dim=1)       # numerator      
                neg_key_matrix = nn.functional.normalize( \
                    torch.cat([torch.stack(all_pos_emb[i], dim=0), label_token_repr[i,layer,:].unsqueeze(0)], dim=0), dim=1) # denominator
                
                pos_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), pos_key_matrix)
                neg_sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), neg_key_matrix)

                numerator = torch.sum(torch.exp(pos_sim / self.cont_configs.tau))
                denominator = numerator + torch.sum(torch.exp(neg_sim / self.cont_configs.tau))
                neg_dual_cont_loss = -torch.log(numerator / denominator)

                pos_cont_loss = (1-self.cont_configs.neg_part_weight) * pos_dual_cont_loss
                neg_cont_loss = self.cont_configs.neg_part_weight * neg_dual_cont_loss

                if layer_atten is not None:
                    each_layer_cont_loss += (pos_cont_loss+neg_cont_loss) * layer_atten[i][layer] 
                else:
                    each_layer_cont_loss += (pos_cont_loss+neg_cont_loss)

            contrastive_loss += each_layer_cont_loss
        
        contrastive_loss /= label_depth
        
        if self.cont_configs.use_rank:
            ## Rank Loss.
            for i in range(bz):
                for layer, nodes in self.depth2label.items():
                    if layer < label_depth-1 and len(pos_labels[i][layer+1]) > 0 and len(neg_labels[i][layer+1]) > 0 :
                        query_vector = nn.functional.normalize(label_token_repr[i, layer, :], dim=0)
                        key_matrix = nn.functional.normalize(torch.cat([torch.stack(all_layers_pos_emb[i][layer+1], dim=0), \
                                                                            torch.stack(all_layers_neg_emb[i][layer+1], dim=0)], dim=0), dim=1)
                        
                        sim = nn.functional.cosine_similarity(query_vector.unsqueeze(0), key_matrix)
                        cand_ids = all_layers_pos[i][layer+1] + all_layers_neg[i][layer+1]
                        coms = list(itertools.combinations(cand_ids, 2))
                        # nn.MarginRankingLoss()
                        cur_node_children = []
                        for cur_node in all_layers_pos[i][layer]:
                            if cur_node in self.slot2value: 
                                cur_node_children.extend(self.slot2value[cur_node])
                        
                        rank_gts = compute_rank_gt(all_layers_pos[i][layer+1], cur_node_children, coms)
                        x1 = torch.stack([sim[cand_ids.index(c)] for i, (c,_) in enumerate(coms) if rank_gts[i] != 0], dim=0)
                        x2 = torch.stack([sim[cand_ids.index(c)] for i, (_,c) in enumerate(coms) if rank_gts[i] != 0], dim=0)
                        rank_gts = torch.tensor([r for r in rank_gts if r != 0]).sign().to(x1.device)
                        rank_loss += self.rank_loss_func(x1, x2, rank_gts)

            contrastive_loss += (self.cont_configs.rank_loss_weight * rank_loss) / (label_depth-1)
        
        contrastive_loss /= bz

        return contrastive_loss
     

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    
    def get_predict_labels(self, b_prediction_scores, b_top_k):
        b_predict_labels = []
        for i, prediction_scores in enumerate(b_prediction_scores):
            pred_label = []
            for layer, nodes in self.depth2label.items():   
                layer_score = prediction_scores[nodes]  
                sort_score, sort_idx = torch.sort(layer_score, descending=True)
                sort_idx = sort_idx[:b_top_k[i]]
                pred_label.extend([self.depth2label[layer][idx] for idx in sort_idx])
            b_predict_labels.append(pred_label)
        return b_predict_labels


    def calculate_multi_task_loss(self, multiclass_logits, batch_pos_labels, probs_batch):
        """ Label Hierarchy self-sensing module. """
        ## Obtain truth labels.
        ### 1 Obtain the labels of the model Verbalizers.
        pred_labels = self.get_predict_labels(multiclass_logits, b_top_k=[len(b[0]) for b in batch_pos_labels])   # [bz, each_sample_pred_labels_group]
        pred_paths = compose_path(pred_labels, self.slot2value, self.value2slot)
        ### 2 Determining whether the predicted label nodes at each level can form valid label paths
        tgt_1 = [is_all_ones(p) for p in judge_consistency_path(pred_paths, self.path_set)]

        ### 3 determining whether the predicted label paths are completely correct
        tgt_2 = [is_all_ones(p) for p in judge_correct_pred(pred_labels, [list(itertools.chain(*b)) for b in batch_pos_labels])]

        ## Calculate loss.
        tgt = torch.FloatTensor([[t1, t2] for t1, t2 in zip(tgt_1, tgt_2)]).to(probs_batch.device)
        multi_task_loss = nn.BCELoss()(probs_batch, tgt) 
        return multi_task_loss


    def pos_neg_sampling(self, truth_labels, multiclass_logits):
        """ Obtain positive samples and sample a specified number of negative samples. 
        
        Args:
            truth_labels: torch.Size([bz, label_num]), ground truth(Multi label classification form) for batch samples.
            multiclass_logits: torch.Size([bz, label_num]), output logits for batch smaples.
        """

        # 1. obtain positives
        bz = truth_labels.shape[0]
        label_depth = len(self.depth2label)
        batch_pos_labels = [[[] for _ in range(label_depth)] for _ in range(bz)]
        
        for i in range(bz):
            pos_label_ids = torch.where(truth_labels[i] == 1)[0]
            for p in pos_label_ids:
                for k, v in self.depth2label.items():
                    if p.item() in v:
                        batch_pos_labels[i][k].append(p.item())
                        break

        # 2. Negative sampling strategy and quantity setting.
        ## 2.1. random sample -- Completely random without maintaining hierarchical structure
        batch_neg_labels = [[] for _ in range(bz)]     # List, [bz, label_deep, sample_num]
        for i, sample_pos_labels in enumerate(batch_pos_labels):
            # For each sample
            if self.cont_configs.negative_sample_mode == 'random':
                neg_labels = negative_sampling(self.depth2label, sample_pos_labels, sample_num_list=self.cont_configs.negative_num_list, mode='random',)
            elif self.cont_configs.negative_sample_mode == 'hard':
                neg_labels = negative_sampling(self.depth2label, sample_pos_labels, sample_num_list=self.cont_configs.negative_num_list, mode='hard-verb', output_logits=multiclass_logits[i])
            batch_neg_labels[i] = neg_labels
        return batch_pos_labels, batch_neg_labels
        

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            loss_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            # epoch=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        multiclass_pos = input_ids == self.tokenizer.mask_token_id
        concat_char_pos = torch.zeros_like(multiclass_pos)
        for i in range(multiclass_pos.shape[0]):
            for j in range(1, multiclass_pos.shape[1]-1):
                if multiclass_pos[i][j-1] and multiclass_pos[i][j+1]:
                    concat_char_pos[i][j] = True

        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        if self.training and self.mlmloss_weight > 0:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            # MLM Loss of BERT.
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)    # Randomly replace 13.5% of tokens(exclude [v_i] & [PRED]) in the sentence with [MASK].
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985                                                   # Randomly replace 13.5% of tokens(exclude [v_i] & [PRED]) in the sentence with other tokens.
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)       # single_labels: as a real label for MLM tasks，used for computing mlm_loss; -100: ignore_label_index

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
       
        sequence_output = outputs[0]
        cls_emb = sequence_output[:, 0, :]      # [bz, hidden_size]
        prediction_scores = self.cls(sequence_output)
        
        # ------------------ Verbalizer ----
        # torch.Size([4, 512, 768]) --> torch.Size([4(bz), 2(layer_depth), 768])
        outputs_at_mask = sequence_output.masked_select(      
                multiclass_pos.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).view(sequence_output.size(0),-1,sequence_output.size(-1))
        pos_outputs_at_mask, neg_outputs_at_mask = torch.split(outputs_at_mask, split_size_or_sections=len(self.depth2label), dim=1)

        key_vectors = sequence_output.masked_select(      
            concat_char_pos.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).view(sequence_output.size(0),-1,sequence_output.size(-1))
        key_vectors, _ = torch.split(key_vectors, split_size_or_sections=len(self.depth2label)-1, dim=1) # [bz, label_depth-1, dimension]

        logits = []
        for idx in range(self.verbLength):
            label_words_logtis = self.__getattr__(f"verbalizer{idx}").process_outputs(pos_outputs_at_mask[:, idx, :], batch={'loss_ids': loss_ids })  # torch.Size([4, 7])  [bz, label_num_each_layer]
            logits.append(label_words_logtis)      
        multiclass_logits = torch.cat(logits, dim=1)        # [bz, label_num]    [24, 141]

        masked_lm_loss = 0
        total_loss = 0

        if labels is not None:
                
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            if self.mlmloss_weight > 0:
                masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
           
            truth_labels = []
            trans_labels = labels.view(input_ids.shape[0], -1 , self.num_labels) # [bz, label_depth, label_num]
            truth_labels = torch.max(trans_labels, dim=1)[0]
        
            multiclass_loss = multilabel_categorical_crossentropy(truth_labels, multiclass_logits)
            
            # ------------------ Hierarchy-aware Peer-label Contrastive Learning  --------------------------

            batch_pos_labels, batch_neg_labels = self.pos_neg_sampling(truth_labels, multiclass_logits, )
            cont_loss = self.calculate_contrastive_loss(batch_pos_labels, 
                                                        batch_neg_labels, 
                                                        neg_outputs_at_mask,
                                                        label_token_repr=neg_outputs_at_mask,
                                                        # layer_atten=atten
                                                        )
            
            # --------     Label Hierarchy Self-sensing Task    -----------
            multi_task_loss = 0
            if self.multitask_loss_weight > 0:
                logits = self.multi_task_mlp(cls_emb)
                probs_batch = nn.Sigmoid()(logits)

                multi_task_loss = self.calculate_multi_task_loss(multiclass_logits, batch_pos_labels, probs_batch)

            # total_loss = self.mlmloss_weight * masked_lm_loss + multiclass_loss + self.cont_configs.loss_weight * cont_loss + multi_task_loss * loss_weight
            total_loss = self.mlmloss_weight * masked_lm_loss + multiclass_loss + self.cont_configs.loss_weight * cont_loss + multi_task_loss * self.multitask_loss_weight

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        ret = MaskedLMOutput(
            loss=total_loss,
            logits=multiclass_logits,
            # hidden_states=outputs.hidden_states,
            hidden_states=pos_outputs_at_mask,
        )
        return ret


    @torch.no_grad()
    def generate(self, input_ids, return_feature=False, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        predict_scores = outputs['logits']
        
        predict_labels = []
        for scores in predict_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                if score > 0:
                    predict_labels[-1].append(i)
        if return_feature:
            return predict_labels, predict_scores, outputs['hidden_states']
        return predict_labels, predict_scores 

