# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import copy
import json
import openai
import os
import pickle
import random
import signal 
import submitit
import time
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm 


# Signal Handlers
def handle_sigusr1(signum, frame):
    print(f'Received {signum}, requeuing job.', flush=True)
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    print(f'Received {signum}, bypassing.', flush=True)
    pass



def codex(configs, dataset, prefixes):
    # model
    openai.api_key = os.getenv("OPENAI_API_KEY")
    def codex_greedy(prompt):
        response = openai.Completion.create(
            engine=configs.engine_name if configs.engine_name is not None else 'davinci-codex',
            prompt=prompt,
            temperature=0,
            max_tokens=920,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=configs.end_template
        )
        return response['choices'][0]['text'], None, None

    def codex_sample(prompt):
        response = openai.Completion.create(
            engine=configs.engine_name if configs.engine_name is not None else 'davinci-codex',
            prompt=prompt,
            temperature=configs.temperature,
            max_tokens=920,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=1,
            stop=configs.end_template
        )
        return response['choices'][0]['text'], response['choices'][0]['logprobs']['tokens'], response['choices'][0]['logprobs']['token_logprobs']

    prompt_prefix = ''.join([configs.prompt_template.format(src=x[0], trg=x[1]) for x in prefixes])

    # save folder
    save_dir = f'{configs.output_path}/seed-{configs.seed}/{configs.n_prompts}-shot/{configs.mode}-{configs.temperature}/'
    os.system(f'mkdir -p {save_dir}')
    # save configs and prefixes
    if configs.rank == 0:
        with open(f'{save_dir}/prefixes.json', 'w') as fout:
            json.dump(prefixes, fout)
            fout.close()
        with open(f'{save_dir}/configs.pkl', 'wb') as fout:
            pickle.dump(configs, fout)
            fout.close()
    ofname = f'{save_dir}/{configs.split}-{configs.rank}.jsonl'
    # load checkpoint
    if os.path.exists(ofname):
        n_processed_examples = len(open(ofname).readlines())
    else:
        n_processed_examples = 0
    pbar = tqdm(dataset)

    with open(ofname, 'a') as fout:
        for i, (src, trg) in enumerate(pbar):
            if i < n_processed_examples:
                continue
            prompt = prompt_prefix + configs.example_template.format(src=src)
            while True:
                try:
                    trg_prediction, tokens, logprobs = codex_greedy(prompt) if configs.mode == 'greedy' else codex_sample(prompt)
                    time.sleep(2)
                    break
                except:
                    print('calling too frequently.. sleeping for 30 secs.', flush=True)
                    time.sleep(30)
            try:
                bleu_score = sentence_bleu([[ch for ch in trg]], [ch for ch in trg_prediction])
            except:
                bleu_score = 0
            print(
                json.dumps(
                    {
                        'prompt': prompt,
                        'src': src,
                        'trg_prediction': trg_prediction,
                        'reference': trg,
                        'tokens': tokens,
                        'logprobs': logprobs,
                        'bleu': bleu_score
                    }
                ),
                file=fout, flush=True
            )
            pbar.set_description(f'Process {configs.rank}')
        fout.close()


def codex_with_info(configs, dataset, prefixes):
    # model
    openai.api_key = os.getenv("OPENAI_API_KEY")
    def codex_greedy(prompt):
        response = openai.Completion.create(
            engine=configs.engine_name if configs.engine_name is not None else 'davinci-codex',
            prompt=prompt,
            temperature=0,
            max_tokens=920,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=configs.end_template
        )
        return response['choices'][0]['text'], None, None

    def codex_sample(prompt):
        response = openai.Completion.create(
            engine=configs.engine_name if configs.engine_name is not None else 'davinci-codex',
            prompt=prompt,
            temperature=configs.temperature,
            max_tokens=920,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=1,
            stop=configs.end_template
        )
        return response['choices'][0]['text'], response['choices'][0]['logprobs']['tokens'], response['choices'][0]['logprobs']['token_logprobs']

    prompt_prefix = ''.join([configs.prompt_template.format(src=x[0], trg=x[1], info=x[2]) for x in prefixes])

    # save folder
    save_dir = f'{configs.output_path}/seed-{configs.seed}/{configs.n_prompts}-shot/{configs.mode}-{configs.temperature}/'
    os.system(f'mkdir -p {save_dir}')
    # save configs and prefixes
    if configs.rank == 0:
        with open(f'{save_dir}/prefixes.json', 'w') as fout:
            json.dump(prefixes, fout)
            fout.close()
        with open(f'{save_dir}/configs.pkl', 'wb') as fout:
            pickle.dump(configs, fout)
            fout.close()
    ofname = f'{save_dir}/{configs.split}-{configs.rank}.jsonl'
    # load checkpoint
    if os.path.exists(ofname):
        n_processed_examples = len(open(ofname).readlines())
    else:
        n_processed_examples = 0
    pbar = tqdm(dataset)

    with open(ofname, 'a') as fout:
        for i, (src, trg, info) in enumerate(pbar):
            if i < n_processed_examples:
                continue
            prompt = prompt_prefix + configs.example_template.format(src=src, info=info)
            while True:
                try:
                    trg_prediction, tokens, logprobs = codex_greedy(prompt) if configs.mode == 'greedy' else codex_sample(prompt)
                    time.sleep(2)
                    break
                except Exception as e:
                    print(e, flush=True)
                    time.sleep(30)
            try:
                bleu_score = sentence_bleu([[ch for ch in trg]], [ch for ch in trg_prediction])
            except:
                bleu_score = 0
            print(
                json.dumps(
                    {
                        'prompt': prompt,
                        'src': src,
                        'trg_prediction': trg_prediction,
                        'reference': trg,
                        'tokens': tokens,
                        'logprobs': logprobs,
                        'bleu': bleu_score
                    }
                ),
                file=fout, flush=True
            )
            pbar.set_description(f'Process {configs.rank}')
        fout.close()


""" example collector: <src, trg> """
class Collector(object):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def __call__(self):
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        job_env = submitit.JobEnvironment()
        configs = copy.deepcopy(self.configs)
        configs.rank = job_env.global_rank
        configs.gpu = job_env.local_rank
        configs.world_size = job_env.num_tasks
        for seed in self.configs.seed:
            for n_prompts in self.configs.n_prompts:
                for temperature in self.configs.temperature:
                    configs.n_prompts = n_prompts
                    configs.seed = seed
                    configs.temperature = temperature
                    random.seed(configs.seed)
                    if configs.saved_prefixes_path_template is not None:
                        prefix_pool = list()
                        for path in glob(configs.saved_prefixes_path_template, recursive=True):
                            prefix_pool.extend(json.load(open(path)))
                        prefix_pool = sorted(set([tuple(x) for x in prefix_pool]))
                        prefixes = random.sample(prefix_pool, configs.n_prompts)
                    else:
                        prefixes = random.sample(self.dataset.data['train'], configs.n_prompts)
                    if configs.shuffle_prefix:
                        original_prefixes = copy.deepcopy(prefixes)
                        while original_prefixes == prefixes:
                            random.shuffle(prefixes)
                    codex(configs, self.dataset.data[configs.split], prefixes)
    
    @staticmethod
    def parse_args(main_parser=None):
        if main_parser is None:
            main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(title='commands', dest='mode')
        # collect
        parser = subparsers.add_parser('collect', help='collecting stage')
        parser.add_argument('--output-path', type=str, required=True)
        parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
        parser.add_argument('--seed', type=int, nargs='+', default=[0])
        parser.add_argument('--n-prompts', type=int, nargs='+', default=[3], help='number of few-shot prompt examples')
        parser.add_argument('--mode', type=str, default='greedy', choices=['greedy', 'sample'])
        parser.add_argument('--n-samples', type=int, default=5, help='number of sampled examples under the sampling mode')
        parser.add_argument('--temperature', type=float, default=[0.6], nargs='+', help='sample temperature')
        parser.add_argument('--prompt-template', type=str, default='# {src}\n{trg}\n')
        parser.add_argument('--example-template', type=str, default='# {src}\n')
        parser.add_argument('--end-template', type=str, default='\n')
        parser.add_argument('--shuffle-prefix', action='store_true', default=False)
        parser.add_argument('--saved-prefixes-path-template', type=str, default=None)
        parser.add_argument('--engine-name', type=str, default=None)

        # slurm arguments
        parser.add_argument('--slurm-ntasks', type=int, default=None)
        parser.add_argument('--slurm-ngpus', type=int, default=0)
        parser.add_argument('--slurm-nnodes', type=int, default=1)
        parser.add_argument('--slurm-partition', type=str, default='devlab')

        args = main_parser.parse_args()

        if args.mode == 'greedy': 
            args.n_samples = 1
            args.temperature = [0]
        if args.slurm_ntasks is None:
            args.slurm_ntasks = args.n_samples
        else:
            assert args.slurm_ntasks == args.n_samples
        return args

    @classmethod
    def from_args(cls, args=None, dataset=None):
        if args is None:
            args = cls.parse_args()
        assert dataset is not None
        return cls(args, dataset)


""" example collector: <src, trg, info> """
class CollectorWithInfo(object):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def __call__(self):
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        job_env = submitit.JobEnvironment()
        configs = copy.deepcopy(self.configs)
        configs.rank = job_env.global_rank
        configs.gpu = job_env.local_rank
        configs.world_size = job_env.num_tasks
        for seed in self.configs.seed:
            for n_prompts in self.configs.n_prompts:
                for temperature in self.configs.temperature:
                    configs.n_prompts = n_prompts
                    configs.seed = seed
                    configs.temperature = temperature
                    random.seed(configs.seed)
                    if configs.saved_prefixes_path_template is not None:
                        prefix_pool = list()
                        for path in glob(configs.saved_prefixes_path_template, recursive=True):
                            prefix_pool.extend(json.load(open(path)))
                        prefix_pool = sorted(set([tuple(x) for x in prefix_pool]))
                        prefixes = random.sample(prefix_pool, configs.n_prompts)
                    else:
                        prefixes = random.sample(self.dataset.data['train'], configs.n_prompts)
                    if configs.shuffle_prefix:
                        original_prefixes = copy.deepcopy(prefixes)
                        while original_prefixes == prefixes:
                            random.shuffle(prefixes)
                    codex_with_info(configs, self.dataset.data[configs.split], prefixes)
    
    @staticmethod
    def parse_args(main_parser=None):
        if main_parser is None:
            main_parser = argparse.ArgumentParser()
        subparsers = main_parser.add_subparsers(title='commands', dest='mode')
        # collect
        parser = subparsers.add_parser('collect', help='collecting stage')
        parser.add_argument('--output-path', type=str, required=True)
        parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
        parser.add_argument('--seed', type=int, nargs='+', default=[0])
        parser.add_argument('--n-prompts', type=int, nargs='+', default=[3], help='number of few-shot prompt examples')
        parser.add_argument('--mode', type=str, default='greedy', choices=['greedy', 'sample'])
        parser.add_argument('--n-samples', type=int, default=5, help='number of sampled examples under the sampling mode')
        parser.add_argument('--temperature', type=float, default=[0.6], nargs='+', help='sample temperature')
        parser.add_argument('--prompt-template', type=str, default='<info>{info}</info>\n<text>{src}</text>\n<code>{trg}</code>\n')
        parser.add_argument('--example-template', type=str, default='<info>{info}</info>\n<text>{src}</text>\n<code>')
        parser.add_argument('--end-template', type=str, default='</code>')
        parser.add_argument('--shuffle-prefix', action='store_true', default=False)
        parser.add_argument('--saved-prefixes-path-template', type=str, default=None)
        parser.add_argument('--engine-name', type=str, default=None)
        # slurm arguments
        parser.add_argument('--slurm-ntasks', type=int, default=None)
        parser.add_argument('--slurm-ngpus', type=int, default=0)
        parser.add_argument('--slurm-nnodes', type=int, default=1)
        parser.add_argument('--slurm-partition', type=str, default='devlab')

        args = main_parser.parse_args()

        if args.mode == 'greedy': 
            args.n_samples = 1
            args.temperature = [0]
        if args.slurm_ntasks is None:
            args.slurm_ntasks = args.n_samples
        else:
            assert args.slurm_ntasks == args.n_samples
        return args

    @classmethod
    def from_args(cls, args=None, dataset=None):
        if args is None:
            args = cls.parse_args()
        assert dataset is not None
        return cls(args, dataset)
