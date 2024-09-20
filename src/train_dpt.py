"""
@Desc: Model training entry program of DPT
@Date: 2023-11-30
@Author: xiongsishi@chinatelecom.cn
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import time 

import pickle 
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import datasets
from tqdm import tqdm
import argparse
import wandb

import utils
from evaluation_metrics import evaluate, evaluate_based_on_path

from models.dual_prompt_model import HTCDualPromptModel


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--linear_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--data_dir', type=str, default='data/BGC')
    parser.add_argument('--data_name', type=str, default='bgc')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)          # default=1000)
    parser.add_argument('--early-stop', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str, default='debug') # required=True)
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--arch', type=str, default='bert-base-uncased-bgc')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)

    parser.add_argument('--mlmloss_weight', default=1, type=float)

    # realted to contrastive module
    parser.add_argument('--cont_loss_weight', default=0.1, type=float)
    parser.add_argument('--cont_tau', default=0.5, type=float)
    parser.add_argument('--cont_use_rank', default=1, type=int, choices=[0, 1], help="Whether to use rank loss in contrastive learning module")
    parser.add_argument('--cont_rank_loss_weight', default=1.0, type=float)
    parser.add_argument('--cont_negative_sample_mode', default='hard', type=str, choices=['random', 'hard'])
    parser.add_argument('--cont_negative_num_list', default=[1, 2], type=int, nargs='+')
    parser.add_argument('--cont_neg_part_weight', default=0., type=float)


    # related to multi-task module
    parser.add_argument('--multitask_loss_weight', default=0.5, type=float)
    parser.add_argument('--save_feature', default=0., type=float)

    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict() if not hasattr(self.model, "module") else self.model.module.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)

class ContrastiveHyperameters:
    def __init__(self, tau, loss_weight, negative_sample_mode, negative_num_list, use_rank, rank_loss_weight, neg_part_weight) -> None:
        self.tau = tau
        self.loss_weight = loss_weight
        self.negative_num_list = negative_num_list
        self.negative_sample_mode = negative_sample_mode
        self.use_rank = use_rank
        self.rank_loss_weight = rank_loss_weight
        self.neg_part_weight = neg_part_weight


if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    args.exp_name = args.data_name + '_' + args.exp_name
    args.cont_use_rank = bool(args.cont_use_rank)
    # create save dir
    if not os.path.exists(os.path.join('checkpoints', args.exp_name)):
        os.mkdir(os.path.join('checkpoints', args.exp_name))

    if args.wandb:
        wandb.init(config=args, project='DPT')

    utils.seed_torch(args.seed)
    logger = utils.init_logger(os.path.join('checkpoints', args.exp_name, 'run.log'))
    logger.info(args)

    cont_parameters = ContrastiveHyperameters(tau=args.cont_tau,
                                                loss_weight=args.cont_loss_weight,
                                                negative_sample_mode=args.cont_negative_sample_mode,
                                                negative_num_list=args.cont_negative_num_list,
                                                use_rank=args.cont_use_rank,
                                                rank_loss_weight=args.cont_rank_loss_weight,
                                                neg_part_weight=args.cont_neg_part_weight)


    utils.seed_torch(args.seed)
    if args.device == 'cuda':
        device = torch.device("cuda:0" if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
        logger.info("Use cuda.")

    batch_size = args.batch

    # tokenizer = AutoTokenizer.from_pretrained(args.arch)
    tokenizer = BertTokenizer.from_pretrained(args.arch)
    if os.path.exists(os.path.join(args.data_dir, 'special_tokens.txt')):
        new_tokens = []
        with open(os.path.join(args.data_dir, 'special_tokens.txt'), 'r', encoding='utf8') as f:
            for line in f:
                new_tokens.append(line.strip())
        tokenizer.add_tokens(new_tokens, special_tokens=True)
        print('add special tokens.')

    label_dict = torch.load(os.path.join(args.data_dir, 'value_dict.pt'))      # id2label dict, {0: 'CS', 1: 'Medical'} # flat
    label_dict = {i: v for i, v in label_dict.items()}

    slot2value = dict(torch.load(os.path.join(args.data_dir, 'slot.pt')))         # key - label id of high class level; value(set) - children label ids. e.g. {0: {99, 101, 37, 7, 41, 12, 140, 46, 110, 111, 49, 50, 23, 57, ...}, ...}
    value2slot = {}      # key - label id; value - parent label id. E.g. {99: 0, 101: 0, 37: 0, 7: 0, 41: 0, ...}
    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
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
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}    # key: layer level, value: label ids at the layer. E.g, {0: [0, 1, 2, 3, 4, 5, 6], 1: [7, 8, 9, 10, 11, 12, 13, 14, 15, ...]}

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    # all tree-path
    path_set = utils.get_path_set(slot2value, value2slot, label_dict)

    logger.info('num_class: {}'.format(num_class))
    logger.info('label dict: {}'.format(label_dict))
    logger.info('slot2value: {}'.format(slot2value))
    logger.info('value2slot: {}'.format(value2slot))
    logger.info('depth2label: {}'.format(depth2label))
    logger.info('path_list: {}'.format(path_list))


    if os.path.exists(os.path.join(args.data_dir, 'DPT')):
        dataset = datasets.load_from_disk(os.path.join(args.data_dir, 'DPT'))
    else:
        dataset = datasets.load_dataset('json',
                                        data_files={'train': '{}/{}_train.json'.format(args.data_dir, args.data_name),
                                                    'dev': '{}/{}_dev.json'.format(args.data_dir, args.data_name),
                                                    'test': '{}/{}_test.json'.format(args.data_dir, args.data_name) 
                                                    })    # Default cache exists in: ~/.cache/huggingface/datasets/json/


        template = 'it belongs to'
        prefix = [tokenizer.convert_tokens_to_ids(t) for t in template.split(' ')]
        for i in range(max_depth):
            prefix.append(tokenizer.mask_token_id)
            if i < max_depth-1:
                prefix.append(tokenizer.convert_tokens_to_ids('L-L'))     
                # prefix.append(tokenizer.convert_tokens_to_ids('-'))

        template = 'rather than'
        prefix.extend([tokenizer.convert_tokens_to_ids(t) for t in template.split(' ')])
        for i in range(max_depth):
            prefix.append(tokenizer.mask_token_id)
            if i < max_depth-1:
                prefix.append(tokenizer.convert_tokens_to_ids('L-L'))
                # prefix.append(tokenizer.convert_tokens_to_ids('-'))

        prefix.append(tokenizer.sep_token_id)

        def data_map_function(batch, tokenizer):
            new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': [], 'loss_ids': []}
            for l, t in zip(batch['label'], batch['token']):
                new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)])
                for d in range(max_depth):
                    for i in depth2label[d]:
                        new_batch['labels'][-1][d][i] = 0
                    for i in l:
                        if new_batch['labels'][-1][d][i] == 0:
                            new_batch['labels'][-1][d][i] = 1
                new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y]

                tokens = tokenizer(t, truncation=True)

                input_ids = tokens['input_ids'][:-1][:512 - len(prefix)] + prefix
                new_batch['input_ids'].append(input_ids)
                new_batch['input_ids'][-1].extend(
                    [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1])))
                new_batch['attention_mask'].append(
                    tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                new_batch['token_type_ids'].append([0] * 512)

                loss_ids = [0] * 512
                idxs = [idx for idx, token_id in enumerate(input_ids) if token_id == tokenizer.cls_token_id or token_id == tokenizer.sep_token_id]
                for i in idxs:
                    loss_ids[i] = -100
                idxs = [idx for idx, token_id in enumerate(input_ids) if token_id == tokenizer.mask_token_id]
                for i in idxs:
                    loss_ids[i] = 1

                new_batch['loss_ids'].append(loss_ids)

            return new_batch

        dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
        dataset.save_to_disk(os.path.join(args.data_dir, 'DPT'))
    dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'loss_ids'])
    dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'loss_ids'])
    dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels', 'loss_ids'])
    logger.info("train_data num is: {}".format(len(dataset['train'])))
    logger.info("dev_data num is: {}".format(len(dataset['dev'])))
    logger.info("test_data num is: {}".format(len(dataset['test'])))


    model = HTCDualPromptModel.from_pretrained(args.arch,
                                    tokenizer=tokenizer,
                                    mlmloss_weight=args.mlmloss_weight,
                                    multitask_loss_weight=args.multitask_loss_weight,
                                    cont_configs=cont_parameters,
                                    logger=logger,
                                    layer=args.layer,
                                    depth2label=depth2label,
                                    label_dict=label_dict,
                                    value2slot=value2slot,
                                    slot2value=slot2value)
    # model.init_embedding()
    logger.info(model)
    logger.info(f"Total params: {sum(param.numel() for param in model.parameters()) / 1000000.0}M. ")

    if args.wandb:
        wandb.watch(model)

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=20, num_training_steps=len(train) // args.update * args.epochs)


    if args.device=='cuda' and torch.cuda.device_count() > 1:
        logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)


    # save = Save(model, optimizer, scheduler, args)
    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    update_step = 0
    loss = 0


    start_time = time.time()
    for epoch in range(args.epochs):
        logger.info("------------ epoch {} ------------".format(epoch + 1))
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            logger.info("Early stop!")
            break

        model.train()
        loss_record = []
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = model(**batch)

                step_loss = output['loss']
                if torch.cuda.device_count() > 1:
                    step_loss = torch.mean(step_loss)
                step_loss.backward()
                loss += step_loss.item()
                update_step += 1
                if update_step % args.update == 0:
                    if args.wandb:
                        wandb.log({'loss': loss, })
                    p_bar.set_description(
                        'loss:{:.4f}'.format(loss, ))

                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()
                    loss_record.append(loss)
                    loss = 0
                    update_step = 0
                    # torch.cuda.empty_cache()
        logger.info(
                    f"cur epoch: {epoch}/{args.epochs} \t lr: {scheduler.get_lr()}, \ttrain loss: {sum(loss_record)/len(loss_record)}")

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if hasattr(model, "module"):
                    label_ids, logits = model.module.generate(batch['input_ids'], depth2label=depth2label, )
                else:
                    label_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(label_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        logger.info('macro: {}, micro: {}'.format(macro_f1, micro_f1))
        if args.wandb:
            wandb.log({'val_macro': macro_f1, 'val_micro': micro_f1})
        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.exp_name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.exp_name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        # save(macro_f1, best_score, os.path.join('checkpoints', args.exp_name, 'checkpoint_{:d}.pt'.format(epoch)))
        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.exp_name, 'checkpoint_last.pt'))
        if args.wandb:
            wandb.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()
   
    end_time = time.time()
    logger.info("start_time: ", start_time)
    logger.info("end_time: ", end_time)
    logger.info(f"Training time consuming: {end_time - start_time}" )


    # test
    # test = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    test = DataLoader(dataset['test'], batch_size=256, shuffle=False)
    

    def test_function(extra, save_feature=False):
        model.eval()
        checkpoint = torch.load(os.path.join('checkpoints', args.exp_name, 'checkpoint_best{}.pt'.format(extra)),
                                map_location='cpu')
        logger.info(f'Test load checkpoint: {checkpoint}')
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint['param'])
        else:
            model.load_state_dict(checkpoint['param'])

        pred = []
        gold = []
        res_data = []

        with torch.no_grad(), tqdm(test) as pbar:
            start_time = time.time()
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                if hasattr(model, "module"):
                    if save_feature:
                        label_ids, logits, feature = model.module.generate(batch['input_ids'], depth2label=depth2label, return_feature=False)
                    else:
                        label_ids, logits = model.module.generate(batch['input_ids'], depth2label=depth2label, return_feature=False)
                else:
                    if save_feature:
                        label_ids, logits, feature = model.generate(batch['input_ids'], depth2label=depth2label, return_feature=False)
                    else:
                        label_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, return_feature=False)
                
                batch_gold = []
                for out, g in zip(label_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    batch_gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                batch_gold[-1].append(i)
                gold.extend(batch_gold)

                
                if save_feature:
                    for p, g, f in zip(label_ids, batch_gold, feature):
                        res_data.append({
                            'golds': g,
                            'preds': p,
                            'label_slot_feature':f.detach().cpu().numpy()  
                        })
            if save_feature:
                with open(os.path.join('checkpoints', args.exp_name, 'testdata_dict{}.pkl'.format(extra)), 'wb') as f:
                    pickle.dump(res_data, f)

        end_time = time.time()
        print("Number of test samples: ", len(dataset['dev']))
        print("test phase time consuming: ", end_time - start_time)
        scores = evaluate(pred, gold, label_dict, result_save_dir=os.path.join('checkpoints', args.exp_name), save_flag=extra)
        precision = scores['precision']
        recall = scores['recall']
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        logger.info('macro: {}, micro: {}, precision: {}, recall: {}'.format(macro_f1, micro_f1, precision, recall))
        if args.wandb:
            wandb.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})

        scores = evaluate_based_on_path(pred, gold, label_dict, value2slot, slot2value, result_save_dir=os.path.join('checkpoints', args.exp_name), save_flag=extra)
        c_micro_f1 = scores['c_micro_f1']
        c_macro_f1 = scores['c_macro_f1']
        logger.info('c_micro_f1: {}, c_macro_f1: {}'.format(c_micro_f1, c_macro_f1))

        with open(os.path.join('checkpoints', args.exp_name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, 'precision', precision, 'recall', recall,  'c_micro_f1', c_micro_f1,  'c_macro_f1', c_macro_f1, file=f)

    test_function('_macro', bool(args.save_feature))
    test_function('_micro', bool(args.save_feature))
    
