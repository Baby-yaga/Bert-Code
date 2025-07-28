# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:36:37 2025

@author: ljh93
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
import warnings
warnings.filterwarnings('ignore') # 取消警告信息
import sys
sys.setrecursionlimit(3000) # 递归深度
sys.path.append("E:/气候政策不确定性感知程度/")

def read_data(data_dir):
    def look(content):
        if content == "[]":
            return 0
        else:
            return 1
    data = pd.read_csv(data_dir, sep = "|")
    data['text'] = data['经营(业务)范围']
    data["label"] = data["weak_labels"].progress_apply(lambda x:look(x))
    return data

def fill_paddings(data, maxlen):
    '''补全句长'''
    if len(data) < maxlen:
        pad_len = maxlen-len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:maxlen])
    return data

class InputDataSet():

    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, item):  # item是索引 用来取数据
        text = str(self.data['text'][item])
        labels = self.data['label'][item]
        labels = torch.tensor(labels, dtype=torch.long)

        ## 手动构建
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        input_ids = fill_paddings(tokens_ids,self.max_len)

        attention_mask = [1 for _ in range(len(tokens_ids))]
        attention_mask = fill_paddings(attention_mask,self.max_len)

        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)

        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'labels':labels
        }


if __name__ == '__main__':
    model_dir = 'google-bert/bert-base-chinese'