# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import json
import os
import regex


class NL2BashDataset(object):
    def __init__(self, path='data/nl2bash/data/bash'):
        self.data = collections.defaultdict()
        for split in ['train', 'dev', 'test']:
            nls = [x.strip() for x in open(os.path.join(path, f'{split}.nl.filtered'))]
            cms = [x.strip() for x in open(os.path.join(path, f'{split}.cm.filtered'))]
            self.data[split] = list(zip(nls, cms))


class SpiderDataset(object):
    def __init__(self, path='data/spider'):
        self.data = collections.defaultdict()
        self.dbs = json.load(open(f'{path}/tables.json'))
        self.id2db = {item['db_id']: item for item in self.dbs}
        for split in ['train', 'dev']:
            split_fname = 'train_spider' if split == 'train' else split
            data = json.load(open(f'{path}/{split_fname}.json'))
            nls = [x['question'] for x in data]
            cms = [x['query'] for x in data]
            db_info = [self.extract_db_info(x['db_id']) for x in data]
            self.data[split] = list(zip(nls, cms, db_info))

    def extract_db_info(self, db_id):
        db = self.id2db[db_id]
        id2table = {i: table_name for i, table_name in enumerate(db['table_names_original'])}
        info = f'{db_id} '
        used_table_id = set()
        for table_id, column_name in db['column_names_original']:
            if table_id == -1:
                info += f'| {column_name} '
            elif table_id not in used_table_id:
                info += f'| {id2table[table_id]} : {column_name} '
                used_table_id.add(table_id)
            else:
                info += f', {column_name} '
        return info.strip()


class MBPPGoogleDataset(object):
    def __init__(self, path='data/mbpp/mbpp.jsonl', mode='function_name'):
        raw_data = sorted([json.loads(x) for x in open(path)], key=lambda x: x['task_id'])
        for i, data_item in enumerate(raw_data):
            assert data_item['task_id'] == i + 1
        self.raw_data = collections.defaultdict()
        self.mode = mode
        # 374 for training, 100 heldout, 500 test
        self.raw_data['train'] = raw_data[:10] + raw_data[510:]
        self.raw_data['test'] = raw_data[10:510]
        # data for codex collector, in input-output-info format
        self.data = collections.defaultdict()
        for split in self.raw_data:
            self.data[split] = self.extract_data(self.raw_data[split], mode)

    @staticmethod
    def extract_data(raw_data, mode):
        if mode == 'function_name':
            get_function_name = lambda test_example: regex.match('assert [\(]*([^\(]+)\(', test_example).group(1)
            info = [get_function_name(x['test_list'][0]) for x in raw_data]
        elif mode == 'assertion':
            info = [x['test_list'][0] for x in raw_data]
        elif mode == 'assertion-full':
            info = [x['test_list'] for x in raw_data]
        else:
            raise Exception(f'Mode {mode} not supported.')
        nls = [x['text'] for x in raw_data]
        codes = [x['code'] for x in raw_data]
        return list(zip(nls, codes, info))
