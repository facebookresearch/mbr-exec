# Copyright (c) Meta Platforms, Inc. and affiliates.

import bashlex
import collections
import json
import pickle
import numpy as np
import os
import random
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from evaluate import evaluate_charbleu, evaluate_google_mbpp, evaluate_spider
from execution import execute_mbpp_google_folder, execute_spider_folder, simulate_bash_exec

""" k sets of configs in separate paths, n * k choose 1 selector """
class MultiSampleSelector(object):
    def __init__(self, paths, split='dev'):
        self.paths = list(sorted(glob(paths, recursive=True))) if isinstance(paths, str) else list(sorted(paths))
        self.split = split
        self.data = collections.defaultdict(list)
        self.args = collections.defaultdict(list)
        for i, path in enumerate(self.paths):
            self.args[i] = pickle.load(open(f'{self.paths[0]}/configs.pkl', 'rb'))
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.jsonl'):
                self.data[i, idx].extend([json.loads(x) for x in open(f'{path}/{split}-{idx}.jsonl')])
                idx += 1
        for path_id, sample_id in self.data:
            for item in self.data[path_id, sample_id]:
                try:
                    avg_logprob, sum_logprob = self.extract_logprob_stats(item, path_id)
                    item['avg_logprob'] = avg_logprob
                    item['sum_logprob'] = sum_logprob
                except:
                    item['avg_logprob'] = item['sum_logprob'] = 0
                if self.paths[path_id].find('nl2bash') != -1:  # NL2bash data, exec simulation
                    try:
                        bashlex.parse(item['trg_prediction'])
                        item['executable'] = True
                    except:
                        item['executable'] = False
                    try:
                        item['trg_prediction_splitted'] = simulate_bash_exec(item['trg_prediction'])
                        item['execution_result_simulated'] = collections.Counter(item['trg_prediction_splitted'])
                    except:
                        item['trg_prediction_splitted'] = list()
                        item['execution_result_simulated'] = collections.Counter()

    def extract_logprob_stats(self, item, path_id):
        current_seq = ''
        extracted_position = None
        for i, _ in enumerate(item['tokens']):
            current_seq += item['tokens'][i]
            if current_seq.find(item['trg_prediction']) != -1 and current_seq.find(self.args[path_id].end_template) != -1:
                extracted_position = i + 1
                break
        logprobs = item['logprobs'][:extracted_position] if extracted_position is not None else item['logprobs']
        logprobs = list(filter(lambda x: x<0, logprobs))  # handle potential codex bug on positive log probability
        return np.mean(logprobs), np.sum(logprobs)

    def select(self, ids=None, key_extractor=lambda x:x['avg_logprob'],return_keys=False):
        if ids is None:
            ids = self.data.keys()
        ids = list(sorted(ids))
        print(f'Selecting Samples from IDs: {ids}', flush=True)
        n_examples = len(self.data[ids[0]])
        selected_examples = list()
        sample_keys = collections.defaultdict(list)
        for i in range(n_examples):
            max_key = None
            selected_item = None
            for idx in ids:
                item = self.data[idx][i]
                key = key_extractor(item)
                sample_keys[idx].append(key)
                if max_key is None or key > max_key:
                    max_key = key
                    selected_item = item
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples


class ExecutionBasedMultiSampleSelector(MultiSampleSelector):
    def __init__(self, paths, split='dev', execution_type=None):
        super().__init__(paths, split=split)
        self.execution_type = execution_type
        for i, path in enumerate(self.paths):
            if execution_type == 'mbpp':
                execute_mbpp_google_folder(path)
            elif execution_type == 'spider':
                execute_spider_folder(path)
            else:
                raise Exception(f'Execution type {execution_type} not supported.')
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.exec.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.exec.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result'] = execution_result
                idx += 1
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.execfull.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.execfull.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result_full'] = execution_result
                idx += 1
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.execfullpass.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.execfullpass.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result_full_pass'] = execution_result
                idx += 1


class IntraMultiSampleSelector(MultiSampleSelector):
    def __init__(self, paths, split='dev'):
        super().__init__(paths, split=split)
    
    def select(
                self, 
                ids=None, 
                key_extractor=None, 
                second_key_extractor=None,
                return_keys=False
            ):
        if ids is None:
            ids = self.data.keys()
        elif isinstance(ids, int):
            ids = [(i, j) for i in set(x[0] for x in self.data.keys()) for j in range(ids)]
        ids = list(sorted(ids))
        id_set = set(ids)
        sample_keys = collections.defaultdict(list)
        print(f'Selecting Samples from IDs: {ids}')
        n_examples = len(self.data[ids[0]])
        selected_examples = list()
        for i in range(n_examples):
            max_key = None
            selected_item = None
            for idx in id_set:
                item = self.data[idx][i]
                first_keys = list()
                for grndtruth_idx in ids:
                    grndtruth_item = self.data[grndtruth_idx][i]
                    key = key_extractor(item, grndtruth_item)
                    first_keys.append(key)
                first_key = sum(first_keys)
                second_key = second_key_extractor(item) if second_key_extractor is not None else 0
                current_key = (first_key, second_key)
                item['mbr_key'] = current_key
                sample_keys[idx].append(current_key)
                if max_key is None or current_key > max_key:
                    max_key = current_key
                    selected_item = item
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples


class ExecutionBasedIntraMultiSampleSelector(IntraMultiSampleSelector):
    def __init__(self, paths, split='dev', execution_type=None):
        super().__init__(paths, split=split)
        self.execution_type = execution_type
        for i, path in enumerate(self.paths):
            if execution_type == 'mbpp':
                execute_mbpp_google_folder(path)
            elif execution_type == 'spider':
                execute_spider_folder(path)
            else:
                raise Exception(f'Execution type {execution_type} not supported.')
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.exec.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.exec.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result'] = execution_result
                idx += 1
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.execfull.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.execfull.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result_full'] = execution_result
                idx += 1
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.exec.codexcases.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.exec.codexcases.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result_codexexec'] = execution_result
                idx += 1
            idx = 0
            while os.path.exists(f'{path}/{split}-{idx}.execfullpass.pkl'):
                for j, execution_result in enumerate(pickle.load(open(f'{path}/{split}-{idx}.execfullpass.pkl', 'rb'))):
                    self.data[i, idx][j]['execution_result_full_pass'] = execution_result
                idx += 1


"""equivalence checking functions"""
# base equavalence checking function
def single_exec_result_matching(exec_x, exec_y, good_execution_result): 
    try:
        if exec_x[0] == good_execution_result and exec_y[0] == good_execution_result and exec_x[1] == exec_y[1]:
            return 1
        else:
            return 0
    except:
        return 0


# first assertion call matching
def execution_selection_function(x, y, good_execution_result=0):
    exec_x, exec_y = x['execution_result'], y['execution_result']
    return single_exec_result_matching(exec_x, exec_y, good_execution_result)


# just executability checking
def executability_selection_function(x, good_execution_result=0):
    exec_res = x['execution_result']
    return exec_res[0] == good_execution_result


def bleu_selection_function(x, y):
    return sentence_bleu([[ch for ch in x['trg_prediction']]], [ch for ch in y['trg_prediction']])


def token_bleu_selection_function(x, y):
    return sentence_bleu([x['trg_prediction'].split()], y['trg_prediction'].split())


def bash_execution_tokenbleu_selection_function(x, y):
    if not x['executable'] or not y['executable']:
        return 0
    x = x['trg_prediction_splitted']
    y = y['trg_prediction_splitted']
    return sentence_bleu([x], y)


"""
    select and evaluate a group in batch
    required keys:
        data_split: 'train', 'dev' or 'test'
        temperature: 0.1 .. 1.0
        criterion: 'mbr_exec' ... see full options in the function
        data_path: root data path for the task
        n_samples: number of candidates
        rand_seed: random seed for one experiment
"""
def select_mbpp(args, return_selected=False, return_selector=False):
    data_split, temperature, criterion, data_path, n_samples, rand_seed = args
    mbpp_good_execution_result = 0
    data_path = f'{data_path}/seed-*/**/*-{temperature}/'
    secondary_key_function = None
    if criterion == 'mbr_exec':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'mbpp')
        sample_selection_function = lambda x, y: execution_selection_function(x, y, mbpp_good_execution_result)
        secondary_key_function = lambda x: x['sum_logprob']
    elif criterion == 'logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'mbpp')  # pre-execution for faster evaluation
        sample_selection_function = lambda x: x['sum_logprob']
    elif criterion == 'avg_logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'mbpp')  # pre-execution for faster evaluation
        sample_selection_function = lambda x: x['avg_logprob']
    elif criterion == 'mbr_bleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'mbpp') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: bleu_selection_function(x, y)
    elif criterion == 'mbr_tokenbleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'mbpp') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y)
    elif criterion == 'executability-logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'mbpp')
        sample_selection_function = lambda x: (executability_selection_function(x, mbpp_good_execution_result), x['sum_logprob'])
    elif criterion == 'executability-avglogprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'mbpp')
        sample_selection_function = lambda x: (executability_selection_function(x, mbpp_good_execution_result), x['avg_logprob'])
    elif criterion == 'executability-mbr_bleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'mbpp') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: bleu_selection_function(x, y) * (1 - x['execution_result'][0]) * \
            (1 - y['execution_result'][0])
    elif criterion == 'executability-mbr_tokenbleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'mbpp') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y) * (1 - x['execution_result'][0]) * \
            (1 - y['execution_result'][0])
    else:
        raise ValueError(f'Unknown criterion: {criterion}')
    id_keys = list(selector.data.keys())
    random.seed(rand_seed)
    ids = random.sample(id_keys, n_samples)
    if secondary_key_function is not None:
        selected = selector.select(ids, sample_selection_function, secondary_key_function)
    else:
        selected = selector.select(ids, sample_selection_function)
    if return_selector:
        return selector
    elif return_selected:
        return selected
    else:
        result = evaluate_google_mbpp(selected, 'data/mbpp/mbpp.jsonl', 'test')
        return result


def select_nl2bash(args, return_selected=False, return_selector=False):
    data_split, temperature, criterion, data_path, n_samples, rand_seed = args
    data_path = f'{data_path}/seed-*/**/*-{temperature}/'
    secondary_key_function = None
    if criterion == 'mbr_bleu':
        selector = IntraMultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x, y: bleu_selection_function(x, y)
    elif criterion == 'mbr_tokenbleu':
        selector = IntraMultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y)
    elif criterion == 'mbr_exec_tokenbleu':
        selector = IntraMultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x, y: bash_execution_tokenbleu_selection_function(x, y)
        secondary_key_function = lambda x: x['sum_logprob']
    elif criterion == 'logprob':
        selector = MultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x: x['sum_logprob']
    elif criterion == 'avg_logprob':
        selector = MultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x: x['avg_logprob']
    elif criterion == 'executability-logprob':
        selector = MultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x: (x['executable'], x['sum_logprob'])
    elif criterion == 'executability-avglogprob':
        selector = MultiSampleSelector(data_path, data_split)
        sample_selection_function = lambda x: (x['executable'], x['avg_logprob'])
    else:
        raise ValueError(f'Unknown criterion: {criterion}')
    id_keys = list(selector.data.keys())
    random.seed(rand_seed)
    ids = random.sample(id_keys, n_samples)
    if secondary_key_function is not None:
        selected = selector.select(ids, sample_selection_function, secondary_key_function)
    else:
        selected = selector.select(ids, sample_selection_function)
    if return_selector:
        return selector
    elif return_selected:
        return selected
    else:
        result = evaluate_charbleu(selected)
        return result


def select_spider(
            args, 
            return_selected=False, 
            return_selector=False,
        ):
    data_split, temperature, criterion, data_path, n_samples, rand_seed = args
    spider_good_execution_result = True
    data_path = f'{data_path}/seed-*/**/*-{temperature}/'
    secondary_key_function = None
    if criterion == 'mbr_exec':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'spider')
        sample_selection_function = lambda x, y: execution_selection_function(x, y, spider_good_execution_result)
        secondary_key_function = lambda x: x['sum_logprob']
    elif criterion == 'logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'spider')  # pre-execution for faster evaluation
        sample_selection_function = lambda x: x['sum_logprob']
    elif criterion == 'avg_logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'spider')  # pre-execution for faster evaluation
        sample_selection_function = lambda x: x['avg_logprob']
    elif criterion == 'mbr_bleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'spider') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: bleu_selection_function(x, y)
    elif criterion == 'mbr_tokenbleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'spider') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y)
    elif criterion == 'executability-logprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'spider')
        sample_selection_function = lambda x: (executability_selection_function(x, spider_good_execution_result), x['sum_logprob'])
    elif criterion == 'executability-avglogprob':
        selector = ExecutionBasedMultiSampleSelector(data_path, data_split, 'spider')
        sample_selection_function = lambda x: (executability_selection_function(x, spider_good_execution_result), x['avg_logprob'])
    elif criterion == 'executability-mbr_bleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'spider') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: bleu_selection_function(x, y) * x['execution_result'][0] * y['execution_result'][0]
    elif criterion == 'executability-mbr_tokenbleu':
        selector = ExecutionBasedIntraMultiSampleSelector(data_path, data_split, 'spider') # pre-execution for faster evaluation
        sample_selection_function = lambda x, y: token_bleu_selection_function(x, y) * x['execution_result'][0] * y['execution_result'][0]
    else:
        raise ValueError(f'Unknown criterion: {criterion}')
    id_keys = list(selector.data.keys())
    random.seed(rand_seed)
    ids = random.sample(id_keys, n_samples)
    if secondary_key_function is not None:
        selected = selector.select(ids, sample_selection_function, secondary_key_function)
    else:
        selected = selector.select(ids, sample_selection_function)
    if return_selector:
        return selector
    if return_selected:
        return selected
    else:
        return evaluate_spider(selected, 'data/spider/dev_gold.sql', 'all')
