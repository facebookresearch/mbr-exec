# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import tempfile
from datasets import load_metric 
from tqdm import tqdm 

from data import MBPPGoogleDataset
from execution import Command


""" dataset keys: src, trg_prediction, reference """
def evaluate_charbleu(dataset):
    bleu = load_metric('bleu')
    predictions = [[ch for ch in item['trg_prediction']] for item in dataset]
    references = [[[ch for ch in item['reference']]] for item in dataset]
    return bleu.compute(predictions=predictions, references=references)


""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """
def evaluate_spider(dataset, gold_path, eval_mode='all'):
    tempdir = tempfile.TemporaryDirectory(prefix='spider-')
    pred_path = f'{tempdir.name}/preds.sql'
    result_path = f'{tempdir.name}/results.out'
    with open(pred_path, 'w') as fout:
        for item in dataset:
            print(item['trg_prediction'], file=fout)
        fout.close()
    os.system(
        f'''python ~/codebase/nl2code-expr/v2code/spider_official/evaluation.py --gold {gold_path} \
            --pred {pred_path} \
            --etype {eval_mode} \
            --db ~/data/spider/database/ \
            --table ~/data/spider/tables.json > {result_path}
        '''
    )
    results = open(result_path).readlines()
    tempdir.cleanup()
    return results
    

""" dataset keys: src, trg_prediction, reference (only trg_prediction useful) """
def evaluate_google_mbpp(dataset, reference_path, split='test', timeout=10, return_details=False):
    references = MBPPGoogleDataset(reference_path)
    assert len(dataset) == len(references.raw_data[split])
    tempdir = tempfile.TemporaryDirectory()
    passed_information = list()
    pbar = tqdm(references.raw_data[split])
    for i, item in enumerate(pbar):
        if 'execution_result_full_pass' in dataset[i]:
            passed_information.append(int(all(x[1] == True for x in dataset[i]['execution_result_full_pass'])))
        else:
            test_cases = item['test_list']
            test_setups = item['test_setup_code']
            code = dataset[i]['trg_prediction']
            # write code to file 
            with open(f'{tempdir.name}/code.py', 'w') as fout:
                print(code, file=fout)
                print(test_setups, file=fout)
                for case in test_cases:
                    print(case, file=fout)
                fout.close()
            command = Command(f'python {tempdir.name}/code.py >/dev/null 2>&1')
            execution_result = (command.run(timeout=timeout) == 0)
            passed_information.append(int(execution_result))
        pbar.set_description(f'{sum(passed_information)} out of {i+1} passed.')
    tempdir.cleanup()
    if return_details:
        return passed_information
    else:
        return sum(passed_information) / len(passed_information)
