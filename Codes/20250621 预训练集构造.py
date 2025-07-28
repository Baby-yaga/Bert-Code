# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 10:15:58 2025

@author: ljh93
"""
import pandas as pd
import os
import jieba.analyse
import re
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

def clean_text(text): # 文本预处理函数
    text = str(text).lower()
    return text

def match_products(scope_text, product_keywords_map):
    text = clean_text(scope_text)  # 文本转小写
    matched_products = []
    for product, keywords in product_keywords_map.items():
        # 所有关键词也转小写后进行匹配
        lower_keywords = [kw.lower() for kw in keywords]
        if all(kw in text for kw in lower_keywords):  # 严格“全包含”逻辑
            matched_products.append(product)
    return matched_products

road = "E:/企业数据/工商数据/"
output_road = "E:/企业数据/20250622工商数据-绿色产业识别/"
provs = os.listdir(road)
provs = ['上海','云南','内蒙古','北京','吉林',
         '四川','天津','宁夏','安徽','山东',
         '山西','广东','广西','新疆','江苏',
         '江西','河北','河南','浙江','海南',
         '湖北','湖南','甘肃','福建','西藏',
         '贵州','辽宁','重庆','陕西','青海',
         '黑龙江']
# provs = ['江西','河北','河南','浙江','海南']
# provs = ['湖北','湖南','甘肃','福建','西藏',
#          '贵州','辽宁','重庆','陕西','青海',
#          '黑龙江']
# provs = ['贵州','辽宁','重庆','陕西','青海','黑龙江']
# Step 0
products_path = r"C:\Users\ljh93\Desktop\2025年第三季度工作清单\0 绿色产业链识别"
product_df = pd.read_stata(products_path + "/节能环保清洁产业统计分类_2021版本.dta")
product_df = product_df[product_df["产品和服务索引"] != ""]
product_df = product_df[~product_df["产品和服务索引"].isna()]
products = product_df["产品和服务索引"].tolist()

product_keywords_map = {}
safe_keywords = [
    '设备', '装置', '系统', '组件', '仪器', '控制器',
    '变压器', '电机', '风机', '泵', '锅炉', '开关', '电缆', '柜'
]
# 判断是否为“数值类关键词”的函数
def is_noise_keyword(word):
    return bool(re.search(r'^[-+]?\d+(\.\d+)?%?$', word)) or len(word) <= 1

for product in products:
    # 自动提取关键词
    auto_keywords = jieba.analyse.extract_tags(product, topK=5)
    # 清洗：去掉纯数值/百分比/过短词
    auto_keywords = [kw for kw in auto_keywords if not is_noise_keyword(kw)]
    # 强制保留 safe_keywords 中出现在产品名称中的词
    forced_keywords = [kw for kw in safe_keywords if kw in product]
    # 合并并去重
    all_keywords = list(set(auto_keywords + forced_keywords))
    
    product_keywords_map[product] = all_keywords
    
usecols = ["主体身份代码", "企业机构名称", "统一社会信用代码", '行业门类', '行业代码', "经营(业务)范围"]
for prov in provs:
    # Step 1
    path = road + prov + "/" + os.listdir(road + prov)[0]
    data = pd.read_csv(path + "/01-基本信息.csv",sep = "|"
                                               ,usecols = usecols
                                               ,encoding = "utf-16 LE"
                                               ,on_bad_lines = "skip")
    data = data[~data["经营(业务)范围"].isna()]
    data = data[~data["行业代码"].isna()]
    data = data[~data["行业门类"].isna()]
    data = data[~data["统一社会信用代码"].isna()]
    data = data[~data["企业机构名称"].isna()]

    print(prov)
    data['weak_labels'] = data['经营(业务)范围'].progress_apply(lambda x: match_products(x, product_keywords_map))
    data.to_csv(output_road + "{}-subsample.csv".format(prov), index = 0, sep = "|")
