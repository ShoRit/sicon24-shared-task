'''
A script to run a classification task using the transformers library on transfer evaluation on dialogue data.
'''
import argparse
import logging
import os
import random
import sys
import copy
import torch
import csv
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import defaultdict as ddict
import pandas as pd
from dataloader import *
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import transformers


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def seen_eval(model, loader, device, args, tokenizer):
    y_true, y_pred = [], []

    for data in tqdm(loader):
        utterance_ids       = data["input_ids"].to(device)
        utterance_mask      = data["attention_mask"].to(device)
        labels              = data["answers"].to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids = utterance_ids,
                attention_mask = utterance_mask,
                max_length = 10,
                num_beams = 5,
                do_sample = False
            )
        
        preds = [1 if 'yes' in tokenizer.decode(ids, skip_special_tokens=True).lower().strip() else 0 for ids in output_sequences]

        trues = [1 if 'yes' in tokenizer.decode(ids, skip_special_tokens=True).lower().strip() else 0 for ids in labels]

        y_true.extend(trues)
        y_pred.extend(preds)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, acc



def get_predictions(model, loader, device, tokenizer, lbl2desc, args):

    y_true, y_pred          = [], []

    for data in tqdm(loader):
        utterance_ids       = data["input_ids"].to(device)
        utterance_mask      = data["attention_mask"].to(device)
        labels              = data["answers"].to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids = utterance_ids,
                attention_mask = utterance_mask,
                max_length = 10,
                num_beams = 5,
                do_sample = False
            )
        
        preds = [tokenizer.decode(ids, skip_special_tokens=True).lower().strip() for ids in output_sequences]

        trues = [tokenizer.decode(ids, skip_special_tokens=True).lower().strip() for ids in labels]

        y_true.extend(trues)
        y_pred.extend(preds)

    return y_true, y_pred



def map_model_name(model_name):
    if 'flan-t5' in model_name:
        return f'google/{model_name}'
    return model_name

'''
Process the arguments for the baseline code

'''


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      type=str, default='../data/', help='The input directory')
    parser.add_argument('--fewshot',        type=float, default=-1.0, help='The fraction of data to use')
    parser.add_argument('--mode',           type=str, default='ID', help='The mode to run in')
    parser.add_argument('--task',           type=str, default='persuasion', help='The input directory')
    parser.add_argument('--src_task',       type=str, default='persuasion', help='source dataset')
    parser.add_argument('--tgt_task',       type=str, default='persuasion', help='target dataset')
    parser.add_argument('--turns',          type=int, default=5, help='Past context length')
    parser.add_argument('--model_name',     type=str, default='flan-t5-small', help='The model to use')
    parser.add_argument('--do_train',       type=int, default=1, help='Whether to train the model')
    parser.add_argument('--do_predict',        type=int, default=1, help='Whether to test the model')
    parser.add_argument('--max_seq_len',    type=int, default=512, help='The maximum sequence length')
    parser.add_argument('--batch_size',     type=int, default=64, help='The batch size')
    parser.add_argument('--gpu',            type=str, default='0', help='The gpu to use')
    parser.add_argument('--epochs',         type=int, default=5, help='The number of training epochs')
    parser.add_argument('--seed',           type=int, default=0, help='The random seed')
    parser.add_argument('--learning_rate',  type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='The number of gradient accumulation steps')
    parser.add_argument('--patience',       type=int, default=5, help='The number of patience steps')

    args = parser.parse_args()
    return args

if __name__ =='__main__':
    
    args        = get_arguments()
    
    print(args)

    '''
    mode is used for deciding whether to run the models in an indomain (ID) or transfer (TF) setting

    In the ID setting, the source and target task are the same, and the model is trained and evaluated on the same task, say persuasion

    In the TF setting, the source and target task are different, and the model is trained on the source task and evaluated on the target task, say persuasion -> negotiation

    For all these scenarios, the tasks and datasets share a 1-1 correspondence since each task is associated with only one dataset.
    
    '''

    if args.mode == 'ID':
        args.src_task = args.task
        args.tgt_task = args.task
    else:
        args.task = args.tgt_task
    
    '''
    Important for reproducibility; set the seed for the random number generators
    '''
    seed_everything(args.seed)
    
    '''
    This is the identifiable name for each experiment which is used to save the model and the predictions; this includes 
    (i) the mode, (ii) source task, (iii) target task, (iv) model name,  (v) fewshot, (vi) turns, and  (vii) seed. 

    When the mode is ID, the source and the target tasks are the same. 
    '''

    identifiable_name = f'{args.mode}-{args.src_task}-{args.tgt_task}-{args.model_name}-{args.fewshot}-{args.turns}-{args.seed}'

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    '''
    Loading the source and target data (i.e. the train, validation, and test splits) the label descriptions, the tokenizer, and the model.

    Currently, we use the valid split as the test split, since the test split is not available in the current test phase.

    '''

    train_data  = json.load(open(f'{args.input_dir}/{args.task}-train.json'))
    dev_data    = json.load(open(f'{args.input_dir}/{args.task}-valid.json'))
    test_data   = json.load(open(f'{args.input_dir}/{args.task}-valid.json'))
    lbl2desc    = json.load(open(f'{args.input_dir}/{args.task}-labels.json'))

    tokenizer   = AutoTokenizer.from_pretrained(map_model_name(args.model_name))
    model       = AutoModelForSeq2SeqLM.from_pretrained(map_model_name(args.model_name))


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id    
    
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
        model.config.sep_token_id = model.config.eos_token_id

    print("Loading model...")

    '''
    Loading the data loaders for the train, validation, and test splits.
    '''

    train_loader, dev_loader, test_loader = get_data_loaders(train_data, dev_data, test_data, tokenizer, lbl2desc, args)
    
    
    '''
    In case the mode is TF, i.e. transfer learning, we load the model from the source task and train it on the target task.
    This is relevant to ensure that the checkpoint file exists and the model is loaded from the source task.

    In case the mode is ID, i.e. indomain learning, we train the model on the source task by initializing the model from scratch.

    '''

    loaded_checkpoint_file = f"../ckpts/ID-{args.src_task}-{args.src_task}-{args.model_name}--1.0-{args.turns}-{args.seed}.pt"
    
    if args.mode == 'TF':
        # assert the loaded_checkpoint_file exists
        print(loaded_checkpoint_file)
        assert os.path.exists(loaded_checkpoint_file)
        model.load_state_dict(torch.load(loaded_checkpoint_file))
    model.to(device)

    checkpoint_file         = f"../ckpts/{identifiable_name}.pt"
    best_model              = model

    '''
    Case 1: Training the model. The code ensures that irrespective of the mode, the model is trained on the target task. 
    '''

    if args.do_train == 1:

        #######################################
        # TRAINING LOOP                       #
        #######################################
        print("Setting up training loop...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_p, best_r, best_f1 = 0, 0, 0
        kill_cnt = 0

        for epoch in range(args.epochs):
            print(f"============== TRAINING ON EPOCH {epoch} ==============")
            running_loss = 0.0
            model.train()

            for i, data in enumerate(tqdm(train_loader)):
                # load the data from the data loader
                input_ids           = data["input_ids"].to(device)
                attention_mask      = data["attention_mask"].to(device)    
                labels              = data["answers"].to(device)

                optimizer.zero_grad()
                
                labels              = data["answers"].to(device)
                labels[labels[:, :] == tokenizer.pad_token_id] = -100
                output_dict = model(
                    labels = labels,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                )

                # forward pass
                
                loss = output_dict['loss']
                loss.backward()
                running_loss += loss.item()
                if (i + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()

            # final gradient step if we hadn't handled it already
            if (i + 1) % args.grad_accumulation_steps != 0:
                optimizer.step()
                
            print("============== EVALUATION ==============")

            p_dev, r_dev, f1_dev, acc_dev = seen_eval(model, dev_loader, device, args, tokenizer)
            print(f"Eval data F1: {f1_dev} \t Precision: {p_dev} \t Recall: {r_dev}")

            if f1_dev > best_f1:
                kill_cnt = 0
                best_p, best_r, best_f1 = p_dev, r_dev, f1_dev
                best_model = model
                torch.save(best_model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    torch.save(best_model.state_dict(), checkpoint_file)
                    break
            
            print(f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}")

        torch.save(best_model.state_dict(), checkpoint_file)

        print("============== EVALUATION ON TEST DATA ==============")
        best_model.to(device)
        best_model.eval()
        p_test, r_test, f1_test, acc_test = seen_eval(model, dev_loader, device, args, tokenizer)


    if args.fewshot == 0 and args.do_predict ==1 :

        '''
        Load the model for evaluation on the test data. 

        This is relevant for the case where the model is trained on the source data and evaluated on the target data in a zero-shot setting.

        '''
        model.load_state_dict(torch.load(loaded_checkpoint_file))
        model.to(device)
        model.eval()
        
        print("============== EVALUATION ON TEST DATA ==============")
        print(identifiable_name)
        print(f"Loaded model from {loaded_checkpoint_file}, {args.src_task} -> {args.tgt_task}")
        

        dev_labels,  dev_preds   = get_predictions(model, dev_loader, device=device, tokenizer=tokenizer, lbl2desc  =lbl2desc,  args=args)
        test_labels, test_preds  = get_predictions(model, test_loader, device=device, tokenizer=tokenizer, lbl2desc =lbl2desc,  args=args)

        dev_dict, test_dict = ddict(list), ddict(list)

        for i, (dev_pred, dev_label, dev_inst) in enumerate(zip(dev_preds, dev_labels, dev_data)):
            dev_dict['y_pred'].append(dev_pred)
            dev_dict['y_true'].append(dev_label)
            dev_dict['text'].append(dev_inst['utterance'])
            dev_dict['answer'].append(dev_inst['answer'])
            dev_dict['label'].append(dev_inst['label'])
            
        ## carry out the same for the test set
        for i, (test_pred, test_label, test_inst) in enumerate(zip(test_preds, test_labels, test_data)):
            test_dict['y_pred'].append(test_pred)
            test_dict['y_true'].append(test_label)
            test_dict['text'].append(test_inst['utterance'])
            test_dict['answer'].append(test_inst['answer'])
            test_dict['label'].append(test_inst['label'])
            
        dev_df      = pd.DataFrame(dev_dict)
        test_df     = pd.DataFrame(test_dict)

        dev_df.to_csv(f'../csv_files/{identifiable_name}-dev.csv')
        test_df.to_csv(f'../csv_files/{identifiable_name}-test.csv')



    
        exit(0)

    elif args.do_predict == 1:

        '''
        Load the model for evaluation on the test data. This can be used for both the ID and TF settings, except for the zero-shot setting.
        '''

        print("============== EVALUATION ON TEST DATA ==============")

        model.load_state_dict(torch.load(checkpoint_file))
        model.to(device)
        model.eval()
        
        dev_labels,  dev_preds   = get_predictions(model, dev_loader, device=device, tokenizer=tokenizer, lbl2desc  =lbl2desc,  args=args)
        test_labels, test_preds  = get_predictions(model, test_loader, device=device, tokenizer=tokenizer, lbl2desc =lbl2desc,  args=args)

        dev_dict, test_dict = ddict(list), ddict(list)

        for i, (dev_pred, dev_label, dev_inst) in enumerate(zip(dev_preds, dev_labels, dev_data)):
            dev_dict['y_pred'].append(dev_pred)
            dev_dict['y_true'].append(dev_label)
            dev_dict['text'].append(dev_inst['utterance'])
            dev_dict['answer'].append(dev_inst['answer'])
            dev_dict['label'].append(dev_inst['label'])

        ## carry out the same for the test set
        for i, (test_pred, test_label, test_inst) in enumerate(zip(test_preds, test_labels, test_data)):
            test_dict['y_pred'].append(test_pred)
            test_dict['y_true'].append(test_label)
            test_dict['text'].append(test_inst['utterance'])
            test_dict['answer'].append(test_inst['answer'])
            test_dict['label'].append(test_inst['label'])
            
        dev_df = pd.DataFrame(dev_dict)
        test_df = pd.DataFrame(test_dict)

        dev_df.to_csv(f'../csv_files/{identifiable_name}-dev.csv')
        test_df.to_csv(f'../csv_files/{identifiable_name}-test.csv')
    
        exit(0)
    
    
    
        
        
        
