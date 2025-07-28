# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:42:12 2025

@author: ljh93
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import jieba.analyse
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

road = "E:/企业数据/工商数据/"
input_road = "E:/企业数据/20250622工商数据-绿色产业识别/"
provs = os.listdir(road)
provs = ['上海','云南','内蒙古','北京','吉林',
         '四川','天津','宁夏','安徽','山东',
         '山西','广东','广西','新疆','江苏',
         '江西','河北','河南','浙江','海南',
         '湖北','湖南','甘肃','福建','西藏',
         '贵州','辽宁','重庆','陕西','青海',
         '黑龙江']

train = pd.DataFrame()
dev = pd.DataFrame()
for prov in tqdm(provs):
    # Step 1
    data = pd.read_csv(input_road + "{}-subsample.csv".format(prov), sep = "|")
    # break
    data_train1 = data[data["weak_labels"] == "[]"]
    data_train2 = data[data["weak_labels"] != "[]"]
    
    train1_first = data_train1.sample(int(len(data_train1) * 0.3))
    train2_first = data_train2.sample(int(len(data_train2) * 0.3))
    
    train1, dev1 = train_test_split(train1_first, test_size = 0.5)
    train2, dev2 = train_test_split(train2_first, test_size = 0.5)
    
    train = pd.concat([train,train1])
    train = pd.concat([train,train2])
    
    dev = pd.concat([dev,dev1])
    dev = pd.concat([dev,dev2])
    
output_road = "C:/Users/ljh93/Desktop/2025年第三季度工作清单/0 绿色产业链识别/"
train.to_csv(output_road + "train/train.csv",sep = "|",index = 0)
dev.to_csv(output_road + "test/test.csv",sep = "|",index = 0)
