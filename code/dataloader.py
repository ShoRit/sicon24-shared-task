from collections import defaultdict
import random
from typing import Union
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import numpy as np

# create a Dialogue Dataset class and the corresponding DataLoader


def create_mini_batch(samples):
    utterance_ids   = [s["input_ids"] for s in samples]
    utterance_mask  = [s["attention_mask"] for s in samples]
    answer          = [s["answer"] for s in samples]

    utterance_ids   = pad_sequence(utterance_ids, batch_first=True)
    utterance_mask  = pad_sequence(utterance_mask, batch_first=True)
    answer          = pad_sequence(answer, batch_first=True)

    return {
        "input_ids": utterance_ids,
        "attention_mask": utterance_mask,
        "answers": answer
    }


def sample_fraction(dataset, fraction, task):
    sampled_instances = random.sample(list(enumerate(dataset)), int(fraction * len(dataset)))
    sampled_indices, sampled_dataset = zip(*sampled_instances)
    return sampled_indices, sampled_dataset


def stratified_sample(dataset, n_per_class, task):
    instances_by_label = defaultdict(list)
    for index, instance in enumerate(dataset):
        lbl              = f"{instance['label']}_{instane['answer']}"
        instances_by_label[lbl].append((index, instance))

    all_sampled_instances = []
    for label, instance_list in instances_by_label.items():
        
        if n_per_class > len(instance_list):
            print(
                f"Requested more labels of class {label} ({n_per_class}) than exist ({len(instance_list)}). Using all examples."
            )
            all_sampled_instances.extend(instance_list)
        else:
            sampled_instances = random.sample(instance_list, int(n_per_class))
            all_sampled_instances.extend(sampled_instances)

    random.shuffle(all_sampled_instances)
    sampled_indices, sampled_dataset = zip(*all_sampled_instances)
    return sampled_indices, sampled_dataset


def create_preamble(item, lbl2desc):
    preamble = f"For the task of {item['task']}, the strategy or label {item['label']} {lbl2desc[item['label']]}\n. \
    Given an utterance for the speaker and past dialogue history (wherever available), output 'Yes'\
    if the utterance belongs to the strategy or label {item['label']}, otherwise output 'No'. \
    Your output should contain only 'Yes' or 'No', and no other text.\n."

    return preamble

# describe how the utterance and the labels are meant to be arranged in the dataset

def instruction_LLM(item, tokenizer, lbl2desc, max_len):

    utterance                = item['utterance']
    speaker                  = item['speaker']
    context                  = item['context']
    answer                   = item['answer']
    
    preamble                 = create_preamble(item, lbl2desc)

    conversational_context   = ""
    dialog_text              = f"{preamble}Input:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue]"

    dial_encoding            = tokenizer.encode_plus(
        dialog_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    dial_len                        = len(dial_encoding['input_ids'][0].nonzero())
    ctx_len                         = len(context)
    conversational_context          = ""
    upd_utt_len                     = dial_len

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}"

        curr_utt_encoding            = tokenizer.encode_plus(
            curr_conversational_context,
            add_special_tokens=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        curr_utt_len                = len(curr_utt_encoding['input_ids'][0].nonzero()) 
        
        if upd_utt_len + curr_utt_len  >= max_len: 
            break
        else:
            upd_utt_len             = upd_utt_len + curr_utt_len
            conversational_context  = f'{curr_conversational_context}{conversational_context}'
    
    dialog_text                       = f"{preamble}Input:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue]"

    dial_encoding                   = tokenizer.encode_plus(
        dialog_text + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    answer                         = tokenizer.encode_plus(
        answer + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=5,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )


    return dial_encoding, dialog_text, answer


# process the dialogue dataset with a common format during processing, that has both input as 
# the conversational context and the response which needs to be predicted.

class DialogueDataset(Dataset):
    # needs to be modified later to include the fewshot case as well

    def __init__(self, data, tokenizer, lbl2desc, max_len,  task, fewshot= -1, turns=5):
        
        if fewshot == -1 or fewshot == 0:
            self.data = data
        elif fewshot < 1.0 and fewshot >0:
            sampled_indices, sampled_dataset = sample_fraction(data, fewshot)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)

        elif fewshot > 1.0:
            sampled_indices, sampled_dataset = stratified_sample(data, n_per_class=fewshot, task=task)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)
        else:
            raise AssertionError(
                f"Unexpected value for parameter 'fewshot': {fewshot}. Parameter should be a float in the range (0, 1.0] or an int > 0"
            )
        
        self.tokenizer      = tokenizer
        self.lbl2desc       = lbl2desc
        self.max_len        = max_len
        self.task           = task
        self.turns          = turns
        self.fewshot        = fewshot


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item        = self.data[idx]
        text        = item['utterance']
        
        dial_encoding, dial_text, answer = instruction_LLM(item = item, tokenizer= self.tokenizer, lbl2desc= self.lbl2desc, max_len= self.max_len)
        
    
        # present the text as Instruction followed by input and context and response
        return {
            'text':             text,
            'dialog_text':      dial_text,
            'input_ids':        dial_encoding['input_ids'].flatten(),
            'attention_mask':   dial_encoding['attention_mask'].flatten(),
            'answer':           answer['input_ids'].flatten()
        }


    
def get_data_loaders(
    train_data,
    dev_data,
    test_data,
    tokenizer,
    lbl2desc,
    args,
    shuffle_train=True,
):

    train_set = DialogueDataset(
        train_data,  tokenizer= tokenizer, lbl2desc= lbl2desc, max_len= args.max_seq_len, task = args.task, fewshot = args.fewshot
    )
    
    train_loader = DataLoader(
        train_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=shuffle_train
    )

    dev_set = DialogueDataset(
        dev_data,  tokenizer= tokenizer, lbl2desc= lbl2desc, max_len= args.max_seq_len, task = args.task, fewshot = -1,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = DialogueDataset(
        test_data,  tokenizer= tokenizer, lbl2desc= lbl2desc, max_len= args.max_seq_len, task = args.task, fewshot = -1,
    )
    test_loader = DataLoader(
        test_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )
    

    return train_loader, dev_loader, test_loader

