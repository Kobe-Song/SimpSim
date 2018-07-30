"""
Provides the detailed implementations of all the distance functions.

"""

from time import time
from collections import deque
import numpy as np
import math, logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os

def exec_bfs(G_d_cui, until_layer, workers = 4):
    '''运行BFS算法'''
    futures = {}        # 存储线程
    degreeList = {}     # 存储结点的度序列

    G_cui_d = load_from_disk('cui_d_dict')
    # G_cui_d = load_from_disk('g_d_dict')

    vertices = list(sorted(G_d_cui.keys()))     # 全部疾病结点
    parts = workers            # 一共分成workers部分，分块是为了让CPU并行处理、提高执行效率
    chunks = partition(vertices, parts)     # 把全部顶点分成4部分，chunks存储了workers块被分的顶点

    with ProcessPoolExecutor(max_workers = workers) as executor:
        part = 1
        for c in chunks:
            # 获取块c中的顶点的度序列
            job = executor.submit(getDegreeListsVertices, G_d_cui, G_cui_d, until_layer, c)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()       # 获取进程结果
            degreeList.update(dl)   # 对degreeList进行更新

    # dl = getDegreeListsVertices(G_d_cui, G_cui_d, vertices)
    # degreeList.update(dl)

    logging.info("saving degreeList on dist...")
    save_on_disk(degreeList, 'degreeList')
    logging.info("saving degreeList successfully")

def getDegreeListsVertices(G_d_cui, G_cui_d, until_layer, vertices):
    '''获取图G中所有vertices顶点的度序列'''
    degreeList = {}

    for v in vertices:
        # 获取结点v 在整个图G中前calcUntilLayer邻域内的度序列信息
        degreeList[v] = getDegreeLists(G_d_cui, G_cui_d, until_layer, v)
    return degreeList

def getDegreeLists(G_d_cui, G_cui_d, until_layer, root):
    '''
    获取结点root在G中的度序列

    args:
        G_d_cui: 图G, key: d, value: cui
        G_cui_d: 图G, key: cui, value: d
        root: 起始点
    return:
        listas: 起始点在前calcUntilLayer邻域内, 每个结点的度序列。
    '''
    listas = {}
    vector_access = [0] * (max(G_d_cui) + 1)          # 初始化list, 初始化值为0, 长度为(max(g) + 1)

    queue = deque()     # 存储将要访问的结点
    queue.append(root)  
    vector_access[root] = 1     # 标记结点是否已经访问

    l = deque()         # 存储某一层的度序列

    depth = 0           # 表示结点的第几层邻域
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1     # 表示queue队列中剩余的结点数

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        l.append(len(G_d_cui[vertex]))        # 将顶点的 度 存储到l队列中

        # 以vertex为中心, 存储vertex的邻域
        for u in G_d_cui[vertex]:               # u 为与vertex疾病相关的表型编号
            for v in G_cui_d[u]:                # v 为与表型u相关的疾病编号
                if(vector_access[v] == 0):          # 该疾病编号未被访问过
                    vector_access[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1
        
        # 队列中无结点时, 表示该层结点度数已经计算完成, 存储度序列
        if timeToDepthIncrease == 0:
            lp = np.array(l, dtype='float')
            lp = np.sort(lp)
            listas[depth] = lp                  # 存储第depth层的度序列
            l = deque()

            # OPT3 优化
            if (until_layer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    return listas


def calc_distances_all(vertices, list_vertices, degreeList, part):
    '''利用DTW算法计算度序列的差值，作为两个顶点的距离'''
    distances = {}      # key为结点对, value为该结点对每层的距离(key 为层数, value 为该层距离)
    cont = 0

    # 定义距离计算函数
    dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]       # 结点v1的所有度序列

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]       # 结点v2的所有度序列

            max_layer = min(len(lists_v1), len(lists_v2))
            distances[v1, v2] = {}

            # 利用DTW算法计算度序列的差值
            for layer in range(0, max_layer):
                # 利用DTW求 v1 和 v2 第layer层度序列距离
                dist, path = fastdtw(lists_v1[layer], lists_v2[layer], radius=1, dist=dist_func)
                
                # 衰减因子
                alpha = 0.5
                dist = math.pow(alpha, layer) * dist

                distances[v1, v2][layer] = dist

        cont += 1
    
    preprocess_consolides_distances(distances)      # 对每层距离进行逐层合并
    # save_on_disk(distances, 'distances')
    if part == -1:
        save_on_disk(distances, 'distances-query')
    else:
        save_on_disk(distances, 'distances-' + str(part))
    return


def preprocess_consolides_distances(distances, startLayer=1):
    '''距离合并预处理'''
    logging.info('Consolidating distances...')

    # vertices 为结点对, layers为结点对每层的距离(key为层数, value为该层距离)
    for vertices, layers in distances.items():
        keys_layers = list(sorted(layers.keys()))     # 层数list
        startLayer = min(len(keys_layers), startLayer)  # 开始层
        for layer in range(0, startLayer):
            keys_layers.pop(0)          # 将第一层先出队, 因为后续累加时, 第一层前面为空

        for layer in keys_layers:
            layers[layer] += layers[layer - 1]      # 从初始层数开始, 将每一层距离累加

    logging.info('Distances consolidated.')

# 距离计算函数
def cost(a, b):
    ep = 0.5
    m = max(a, b) + ep
    mi = min(a, b) + ep
    return ((m / mi) - 1)