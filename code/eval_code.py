import os
from glob import glob
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict as ddict
import json

##### evaluation #####


def get_instruct_tuned_results():

    data_dir = '../csv_files/'

    results_dict = ddict(list)
    
    for file in os.listdir(data_dir):

        if 'flan-t5-base' in file:
            model = 'flan-t5-base'
        elif 'flan-t5-small' in file:
            model = 'flan-t5-small'
        elif 'flan-t5-large' in file:
            model = 'flan-t5-large'
        
        elems = file.split('-')
        src_dataset = elems[1]
        tgt_dataset = elems[2]
        dev_or_test = elems[-1].split('.')[0]
        seed        = elems[-2]
        
        data = pd.read_csv(f'{data_dir}/{file}')
        metrics = ddict(lambda: ddict(int))
        global_scores   = {'TP':0, 'FP':0, 'FN':0, 'TN':0}

        for idx, row in data.iterrows():
            # y_pred,y_true,text,answer,label
            gold_lbl       = row['label']
            pred_ans       = 1 if 'yes' in row['y_pred'].lower().strip() else 0
            gold_ans       = 1 if 'yes' in row['y_true'].lower().strip() else 0

            
            assert row['y_true'] == row['answer'].lower().strip()
            

            if pred_ans ==0 and gold_ans ==0:
                metrics[gold_lbl]['TN'] += 1
                global_scores['TN'] += 1
                continue
            else:
                if pred_ans ==1 and gold_ans ==1:
                    metrics[gold_lbl]['TP'] += 1
                    global_scores['TP'] += 1
                elif pred_ans ==1 and gold_ans ==0:
                    metrics[gold_lbl]['FP'] += 1
                    global_scores['FP'] += 1
                elif pred_ans ==0 and gold_ans ==1:
                    metrics[gold_lbl]['FN'] += 1
                    global_scores['FN'] += 1
            
        
        macro_F1        = []

        for lbl in metrics:
            try:
                prec = metrics[lbl]['TP']/(metrics[lbl]['TP'] + metrics[lbl]['FP'])
            except Exception as e:
                prec = 0.0
            
            try:
                rec  = metrics[lbl]['TP']/(metrics[lbl]['TP'] + metrics[lbl]['FN'])
            except Exception as e:
                rec = 0.0
            
            try: 
                f1   = (2*prec*rec)/(prec+rec)
            except Exception as e:
                f1 = 0.0
                
            macro_F1.append(f1)
        
        try:
            micro_rec  = global_scores['TP']/(global_scores['TP'] + global_scores['FN'])
        except Exception as e:
            micro_rec  = 0.0
        
        try:
            micro_prec = global_scores['TP']/(global_scores['TP'] + global_scores['FP'])
        except Exception as e:
            micro_prec = 0.0
        
        try:
            micro_f1 = 2*micro_prec*micro_rec/(micro_prec+micro_rec)
        except Exception as e:
            micro_f1 = 0.0
        
        acc       = (global_scores['TP'] + global_scores['TN'])/(global_scores['TP'] + global_scores['TN'] + global_scores['FP'] + global_scores['FN'])
        
        micro_f1 = round(micro_f1,2)
        acc      = round(acc,2)
        macro_f1 = round(np.mean(macro_F1),2)

        results_dict['model'].append(model)
        results_dict['src_dataset'].append(src_dataset)
        results_dict['tgt_dataset'].append(tgt_dataset)
        results_dict['split'].append(dev_or_test)
        results_dict['seed'].append(seed)
        results_dict['macro_F1'].append(macro_f1)
        results_dict['micro_F1'].append(micro_f1)
        results_dict['acc'].append(acc)

        print(f'{file}\t\t{macro_f1}\t{micro_f1}\t{acc}')

    results_df = pd.DataFrame(results_dict)
    # results_df.to_csv('TF_instruct_tuned_results.csv', index=False)



get_instruct_tuned_results()