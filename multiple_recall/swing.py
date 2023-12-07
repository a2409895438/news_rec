# -*- coding: utf-8 -*-

"""
    Author: lyd
    Created: 2018/11/20
"""

import pandas as pd
from itertools import combinations
import os
import json

def load_data(train_path,test_path):
    train_data = pd.read_csv(train_path, sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])
    test_data = pd.read_csv(test_path, sep="\t", engine="python", names=["userid", "movieid", "rate", "event_timestamp"])

    print(train_data.head(5))
    print(test_data.head(5))
    return train_data, test_data

def get_uitems_iuser(data):
    uitems = dict()
    iusers = dict()
    for i,row in data.iterrows():
        item = row['movieid']
        user = row['userid']
        uitems.setdefault(user,set()).add(item)
        iusers.setdefault(item,set()).add(user)
    return uitems, iusers

def cal_similarity(uitems,iusers,alpha):
    item_sim_dict = dict()
    for (i,j) in list(combinations(iusers.keys(),2)):
        user_pair = list(combinations(iusers[i]&iusers[j],2))
        sim = 0
        for (u1,u2) in user_pair:
            sim += 1/(alpha + len(uitems[u1] & uitems[u2]))
        item_sim_dict.setdefault(i,{})[j] = sim
    return item_sim_dict

def save_item_sim(item_sim_dict,path,k):
    for item in item_sim_dict.keys():
        item_sim_dict[item] = dict(sorted(item_sim_dict[item].items(), key=lambda x:x[1], reverse=True)[:k])
        with open(path+item+".json","w") as f:
            json.dump(item_sim_dict[item],f)
