import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from utils import *
from model import *
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str , help='folder_path',default='../data_path/')
parser.add_argument('--save_path', type=str , default='../tmp_results/')
args = parser.parse_args()



if __name__ == "__main__":
    data_path = args.data_path
    save_path = args.save_path

    # 获取数据
    all_click_df = get_all_click_sample(data_path)

    # 定义
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    user_item_time_dict = get_user_item_time(all_click_df)

    # 去取文章相似度
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))

    # 相似文章的数量
    sim_item_topk = 10

    # 召回文章数量
    recall_item_num = 10

    # 用户热度补全
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    for user in tqdm(all_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,item_topk_click, sim_item_topk=10, recall_item_num=10)

    # 将字典的形式转换成df
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])

    # 获取测试集
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()

    # 从所有的召回数据中将测试集中的用户选出来
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]

    # 生成提交文件
    submit(tst_recall, topk=5, model_name='itemcf_baseline',save_path=save_path)

