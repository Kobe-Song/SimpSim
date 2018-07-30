"""

calculates the meta path based similarity for pairwise nodes


"""

import os
import sys
import time
import xlrd
import xlwt
import pickle
import argparse
from itertools import product
from itertools import combinations


def cal_pathSim(G):
    """
    calculate the meta path-based similarity score for pairwise nodes
    store the results in .pkl file
    :param G:
    :return:
    """

    # # 笛卡尔积
    # l = [1, 2, 3, 4, 5]
    # print(list(product(l, l)))
    # # 排列组合
    # print(list(combinations([1, 2, 3, 4, 5], 2)))

    mimnumber_set = G.keys()  # all disease
    print(len(mimnumber_set))  # all disease mimnumbers

    d_combination_list = list(combinations(mimnumber_set, 2))  # 对全部疾病进行两两排列组合，组合结果存入list中

    print(d_combination_list)

    d_pair_sim_dict = {}  # 存储所有疾病对之间的pathsim值

    for pair in d_combination_list:
        mim1 = pair[0]
        mim2 = pair[1]
        cui_list_1 = G[mim1]
        cui_list_2 = G[mim2]
        intersect_cui_list = list(set(cui_list_1).intersection(set(cui_list_2)))

        path_count_mim1_mim2 = len(intersect_cui_list)  # 分子部分，cui交集个数

        path_count_self_mim1 = len(cui_list_1)  # d1的cui个数
        path_count_self_mim2 = len(cui_list_2)  # d2的cui个数

        # 计算pathsim值
        pathsim_score = (2 * path_count_mim1_mim2) / (path_count_self_mim1 + path_count_self_mim2)

        # 只把相似度大于0的记录保存, 1076370
        if (pathsim_score > 0):
            d_pair_sim_dict[(mim1, mim2)] = pathsim_score
            print("(", mim1, ",", mim2, "), pathsim:", pathsim_score)

    print("pathsim file length:", len(d_pair_sim_dict))

    # 输出为.pkl文件
    reference_relations_file = os.path.join('save/d_pair_pathsim.pkl')

    if os.path.exists(reference_relations_file):
        print("find dumped reference relations file, skip pathsim calculating.")
    else:
        # 存储计算结果
        with open(reference_relations_file, 'wb') as f:
            pickle.dump(d_pair_sim_dict, f)
            print("dump PathSim file successfully!")

    return

