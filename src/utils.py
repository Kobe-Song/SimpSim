"""
Provides some general functions.

"""
import xlrd
import xlwt
import pickle
import argparse
import os
import datetime
import time
from collections import Counter, defaultdict
from itertools import product
from itertools import combinations
import logging

SAVE_FOLDER_NAME = "save/"

def load_from_disk(filename):
    logging.info('load data from disk ...')
    val = None
    
    with open(SAVE_FOLDER_NAME + filename + '.pkl', 'rb') as handle:
        val = pickle.load(handle)
    
    return val

def save_on_disk(data, filename):
    logging.info('save data on disk')

    file_output = os.path.join(SAVE_FOLDER_NAME + filename + '.pkl')

    # if os.path.exists(file_output):
    #     print("find dumped file, skip output.")
    # else:
        # 存储结果文件
    with open(file_output, 'wb') as handle:
        pickle.dump(data, handle)
    
    logging.info('Data has been saved.')

def partition(lst, n):
    division = len(lst) / float(n)  # 总数/分块数 = 每一部分存的元素个数
    # 返回每一部分包含的元素，一共n部分，每一部分大小为division
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]