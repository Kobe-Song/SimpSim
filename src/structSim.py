"""
This file provides all the methods(only entrances) of calculating the structural similarity for pairwise nodes.
All implementations will be found in algorithm_dis.py

"""

import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque

from utils import *
from algorithm_dis import *
import graph



def generate_degreeSeq_with_bfs(G, until_layer, workers = 4):
    '''
    产生结点的度序列
    '''
    logging.info("start to generate degree sequence")

    with ProcessPoolExecutor(max_workers = workers) as executor:  #创建一个进程池，默认容量为1
        job = executor.submit(exec_bfs, G, until_layer, workers)  # 产生1个进程进行工作，获取G中顶点的度序列

        job.result()
    # exec_bfs(G)

    print("finished")
    return

# def generate_degreeSeq_with_bfs_query(G, until_layer,query_d):
#     '''根据疾病编号产生该疾病的度序列'''

#     G_other_d = load_from_disk('other_d_dict')

#     degreeList = {}
#     for i in range(len(query_d)):
#         degreeList[i] = getDegreeLists(G, G_other_d, until_layer,i)

#     save_on_disk(degreeList, 'degreeList-query')
#     print("query finished")

#     return


def calc_distances_all_vertices(G, workers = 4):
    '''计算每个结点之间的距离'''
    logging.info("start to calculate distance")

    print("start to calculate")
    futures = {}

    vertices = list(reversed(sorted(G.keys())))         # 将结点从大到小排序

    degreeList = load_from_disk('degreeList')       # 读取度序列
    
    chunks = partition(vertices, workers)   # 将顶点分成workers块, 并行工作

    with ProcessPoolExecutor(max_workers = workers) as executor:
        part = 1
        for c in chunks:
            logging.info("Executing calc_distances part {}...".format(part))
            list_v = []
            for v in c:         # 从块c中取出每个结点v
                # 依次取出需要与结点v比较的结点list, 避免结点重复比较
                list_v.append([vd for vd in degreeList.keys() if vd > v])
            job = executor.submit(calc_distances_all, c, list_v, degreeList, part)
            futures[job] = part
            part += 1

        logging.info("Receiving results...")

        # 取出计算结果
        for job in as_completed(futures):
            job.result()
            r = futures[job]
            logging.info("Part {} Completed.".format(r))
    
    # list_v = []
    # for v in vertices:
    #     list_v.append([vd for vd in degreeList.keys() if vd > v])
    # calc_distances_all(vertices, list_v, degreeList)

    logging.info("Distances calculated.")

    return


def calc_distances_query(G, query_d):
    '''计算query_d与其他结点之间的距离'''
    vertices = list(reversed(sorted(G.keys())))         # 将结点从大到小排序
    degreeList = load_from_disk('degreeList')       # 读取度序列
    list_v = []
    for v in query_d:
        list_v.append([vd for vd in degreeList.keys() if vd > v])
    calc_distances_all(query_d, list_v, degreeList, part = -1)

    return

def consolide_distances(workers = 4):

    distances = {}

    parts = workers
    for part in range(1, parts + 1):
        d = load_from_disk('distances-' + str(part))
        # preprocess_consolides_distances(distances)
        distances.update(d)


    # preprocess_consolides_distances(distances)
    save_on_disk(distances, 'distances')

    return

def calc_strucSim():
    '''计算结构相似度'''
    struct_sim = {}

    distances = load_from_disk('distances')
    
    
    for vertices, layers in distances.items():
        # 取第k层(最大层)距离, 计算两个结点之间的相似度
        max_layer = max(layers.keys())
        max_distance = layers[max_layer]
        struct_sim[vertices] = math.exp(-max_distance)
        print(vertices, "structsim:", struct_sim[vertices])
    
    out_file = 'd_other_structsim'
    save_on_disk(struct_sim, out_file)

    return out_file