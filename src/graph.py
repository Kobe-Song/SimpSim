# -*- coding: utf-8 -*-

"""Graph utilities."""


import logging
import sys
import math
from io import open
from os import path
from time import time

import xlrd
import xlwt
import pickle
import argparse
import os
import datetime
import time

from utils import *
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations



"""
store nodes(2 types) into a dict:

dict = {
            v1:[u1, u2]
            v2:[u2, u3, u4]
            ...
            vn:[u1, u2, u3, ..., uk]
    
        }

"""

# filename = "data/mim_dname_pid_des_cui.xlsx"

def make_consistent(g):
    '''
    为图中每个结点的邻居结点排序
    args:
        g: 图g
    '''
    for key in g.keys():
        g[key] = list(sorted(set(g[key])))  # 对字典每个键对应的值（是一个list）按从小到大排序
    
    return g

def load_edge_file(filename):

    d_dict = {}  #存储疾病完整信息
    d_p_dict = defaultdict(list)    # 存储编号和表型完整信息
    d_cui_dict = defaultdict(list)  # 存储疾病编号和表型编号
    cui_d_dict = defaultdict(list)  # 存储表型编号和疾病编号
    # d_g_dict = defaultdict(list)
    # g_d_dict = defaultdict(list)

    mimnumber_set = set()  # 用于鉴别每一个疾病，在加入到字典的key中之前，先存入到这个集合中

    data = xlrd.open_workbook(filename)  # 打开xls文件
    # data = load_from_disk(filename, 'xls')
    sheet_num = len(data.sheets())  # 工作表的总数量
    print(filename, " total sheets:", sheet_num)

    for sheet in data.sheets():
        rows = sheet.nrows  # 获取表的行数
        for i in range(rows):
            if i == 0:
                continue  # 跳过第一行的属性名
            else:
                mimnumber = int(sheet.row_values(i)[0])  # 疾病编号
                d_name = sheet.row_values(i)[1]  # 疾病名称
                # g_id = sheet.row_values(i)[1]     # 基因编号
                p_id = sheet.row_values(i)[2]  # 表型编号（自己编的号, 并不规范。其实这个比较多余, 暂时还是先存着）
                p_des = sheet.row_values(i)[3]  # 表型名称
                cui = sheet.row_values(i)[4]  # cui（指代唯一一个表型的本体编号）

                # 添加疾病信息到字典
                d_dict[mimnumber] = d_name

                # 添加疾病对应的全部表型信息
                d_p_dict[mimnumber].append((cui, p_id, p_des))

                d_cui_dict[mimnumber].append(cui)  # 仅包含疾病编号和cui, 后期主要使用这个图字典, key 为疾病编号
                # d_g_dict[mimnumber].append(g_id)

                # print("mimnumber:", mimnumber,",p_name:",p_name,"p_id:",p_id)
                mimnumber_set.add(mimnumber)  # (无实际作用, 用于校验疾病数量)

                cui_d_dict[cui].append(mimnumber)   # 仅包含cui和疾病编号, key 为cui, value 为疾病编号
                # g_d_dict[g_id].append(mimnumber)

    print(len(mimnumber_set))  # 3928
    print("d_dict", len(d_dict))  # 3928
    print("d_p_dict", len(d_p_dict))  # 3928
    print("d_cui_dict:", len(d_cui_dict))  # 3928个
    print("d_cui_dict", len(d_cui_dict[242900])) #68个
    print(d_cui_dict[242900])
    # print("d_g_dict", len(d_g_dict))
    # print("g_d_dict", len(g_d_dict))

    # 输出为.pkl文件
    save_on_disk(d_dict, 'd_dict')
    save_on_disk(d_p_dict, 'd_p_dict')
    save_on_disk(d_cui_dict, 'd_cui_dict')
    save_on_disk(cui_d_dict, 'cui_d_dict')
    # save_on_disk(d_g_dict, 'd_g_dict')
    # save_on_disk(g_d_dict, 'g_d_dict')

    return d_cui_dict
    # return d_g_dict


# load_edge_file(filename)
