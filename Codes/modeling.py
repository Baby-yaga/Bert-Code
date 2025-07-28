# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:35:20 2025

@author: ljh93
"""
from data_process import read_data,InputDataSet
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from tqdm import tqdm
import os
import pandas as pd
tqdm.pandas(desc='pandas bar')

## 做句子的分类 BertForSequence
class BertForSeq(BertPreTrainedModel):

    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 2 # 类别数目
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) # 防止过拟合
        self.classifier = nn.Linear(config.hidden_size, self.num_labels) # 分类器

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
            return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ## loss损失 预测值preds
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )  ## 预测值

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        ## logits -—— softmax层的输入（0.4， 0.6）--- 1
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 二分类任务 这里的参数要做view
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  ##损失
            logits=logits,  ##softmax层的输入，可以理解为是个概率
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def predict(model, text, tokenizer, max_length, device):  
    model.eval()  # 切换到评估模式
    # 对输入文本进行编码    
    encoding = tokenizer(text, 
                         return_tensors='pt', 
                         max_length = max_length,
                         padding = 'max_length', 
                         truncation = True)    
    input_ids = encoding['input_ids'].to(device)    
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    # 获取模型预测    
    with torch.no_grad():    
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        pred = torch.argmax(outputs.logits, dim = 1).cpu().numpy()[0]

    return 1 if pred == 1 else 0

def batch_predict(texts, model, tokenizer, max_length, device, batch_size):
    model.eval()
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts,
                           padding=True,
                           truncation=True,
                           max_length=max_length,
                           return_tensors="pt").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.cpu().numpy().tolist()
            all_preds.extend(preds)
    return all_preds

if __name__ == '__main__':
    ## 设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## 加载编码器和模型
    model_dir = "google-bert/bert-base-chinese"
    model = BertForSeq.from_pretrained(model_dir)
    state_dict = torch.load('C:/Users/ljh93/Desktop/2025年第三季度工作清单/0 绿色产业链识别/result/model_green.bin')
    model.load_state_dict(state_dict)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = model.to(device)
    ## 预测
    road = "E:/企业数据/工商数据/"
    input_road = "E:/企业数据/20240426+工商数据+股权投资/1998-2014年工业企业股权投资/invest-matched.csv"

    data = pd.read_csv(input_road, sep = "|")
    texts = data["经营范围_son"].tolist()
    # 批量预测
    preds = batch_predict(texts, 
                          model, 
                          tokenizer, 
                          max_length = 512, 
                          device = device, 
                          batch_size = 4)
    data["Labels_pred"] = preds
    # data["Labels_pred"] = data["经营范围_son"].progress_apply(lambda x: predict(model, x, tokenizer, 512, device))
    data.to_csv("E:/企业数据/20240426+工商数据+股权投资/1998-2014年工业企业股权投资/green_invest-matched.csv", sep = "|", index = 0)
