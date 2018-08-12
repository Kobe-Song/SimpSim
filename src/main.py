# -*- coding: utf-8 -*-
import sys
import argparse, logging
import numpy as np
from time import time
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from graph import *
from pathsim import *
from structSim import *
from sim2vec import *
from pathsim import *
from combSim import *
from sim_search import *
import utils

logging.basicConfig(filename='SimpSim.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')


def parse_args():
    '''
    Parses the SimpSim arguments.
    '''
    parser = argparse.ArgumentParser(description="Run SimpSim.")

    # 输入文件（节点、边文件）
    parser.add_argument('--input', nargs='?', default='data/simpsim_public_hmdd_d_rna.csv', help='Input graph path')

    parser.add_argument('--output', nargs='?', default='/emb/disease.emb', help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-layer', type=int, default=None, help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int, help='Number of epochs in SGD')  # 重复游走的次数，一个节点执行5次独立的游走，产生5个独立的序列作为这个节点的上下文。

    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers. Default is 4.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool, help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool, help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool, help='optimization 3')
    return parser.parse_args()


def exec_SimpSim(args, filelist):
    """
        对单个数据集计算疾病相似度，得到单视角下的疾病相似网络
        1. read_graph: read file and build graph(HIN, stored by dict)
        2. exec_pathSim: cal meta-path-based similarity for pairwise nodes (represents semantic similarity)
        3. exec_structSim: cal structural similarity for pairwise nodes (represents structural similarity)
        4. combine PathSim and StructSim together and get the final similarity
    """

    #  这里需要写一个循环，从参数列表里面读取一共有多少个文件，依次对每个数据文件计算得到疾病相似网络并输出保存

    multi_sim_network = []  # 存储复合相似网络，二维矩阵存储

    sim_file_list = []  # 存储对各个数据集最终计算得到的相似文件名称

    file_index = 0
    for edgefile in filelist:

        G = read_graph(edgefile)  # 读取边文件，构建图

        pathSim_file_name = exec_pathSim(G)  # 计算语义相似度，返回保存的文件名称

        structSim_file_name = exec_StructSim(G, edgefile)  # 计算结构相似度，返回保存的文件名称
        # pathSim_file_name = 'd_chemical_888d_pair_pathsim'
        # structSim_file_name = 'd_chemical_888d_other_structsim'

        combined_file_name = exec_combine_sim(pathSim_file_name, structSim_file_name, file_index)  # 合并两种相似度并输出最终的相似度【矩阵存储？】【边文件存储？】

        print("dataset:", edgefile, " - calculation of combined similarity score finished!")

        sim_file_list.append(combined_file_name)  # 将对单个数据集计算得到的最终相似结果文件存储到文件中

        file_index += 1

    return sim_file_list


def exec_pathSim(G):
    """
    :param G: original d_p dict
    :return: dict with pathsim score
    """
    # 基础搜索算法
    logging.info(" - Processing pruned PathSim calculation...")

    time_start = time.time()

    pathsim_file_name = cal_pathSim_all(G)  # calculates PathSim score between all pairwise nodes and return the filename that stores the results

    time_end = time.time()
    print('total time cost for pathSim: ', time_end - time_start)

    logging.info(" - PathSim calculation finished.")

    return pathsim_file_name


def exec_StructSim(G, filename):
    '''
    calc structural similarity for all pairwise nodes .

    最后这里也还是将字典输出存入.pkl文件

    '''

    time_start = time.time()
    # 获取疾病结点的度序列
    generate_degreeSeq_with_bfs(G, 2)
    print(filename, "generate degree Sequence finished")

    # 计算结点之间的距离
    calc_distances_all_vertices(G)
    print(filename, "calculate structure distance Sequence finished")

    # 合并每层距离
    consolide_distances()

    # 计算结构相似度
    structSim_file_name = calc_strucSim(filename)

    time_end = time.time()
    print('total time cost for structSim: ', time_end - time_start)
    logging.info(" - StructSim calculation finished.")

    return structSim_file_name


def exec_combine_sim(pathSim_file_name, structSim_file_name, index):
    """
        combine pathsim and structsim together and get the final similarity value for each pairwise nodes.
        here we use inequality equation to combine them
        分别从文件中读出语义相似字典以及结构相似字典，用均值不等式将二者融合，将最终结果输出保存
    """
    time_start = time.time()

    combined_file_name = combine_sim(pathSim_file_name, structSim_file_name, index)

    time_end = time.time()
    print('total time cost for combSim: ', time_end - time_start)

    return combined_file_name


# def exec_simulate_walks(M, num_walks, walk_length):
#     """
#         conduct Random Walk on multiplex network M and generate node sequences for each node as the context
#     """

#     return


def exec_construct_multi_sim_network(sim_file_list):
    """
        将多个相似网络关联在一起，构建一个多层相似网络。在该网络上对每个节点开展随机游走，获得上下文（eg. 迭代游走5次，将产生的5段序列作为该节点的上下文）
        1. associate all the similarity network together, then we get a weighted multiplex similarity network.
        2. for every node, conduct the biased random walk for 5 times to get 5 independent node sequences as the context of this node.
        3. output the walk results onto disk. eg.walk_result.txt
        4. conduct Random Walk on multiplex network M and generate node sequences for each node as the context
    """
    time_start = time.time()
    construct_multi_sim_network(sim_file_list)  # 构建多层网络
    print("construct multiple network finished")
    # preprocess_parameters_random_walk()  # 初始化随机游走参数

    simulate_walks(args.num_walks, args.walk_length)  # 在多层图中随机游走，对每个节点产生上下文
    print("random walk finished")
    time_end = time.time()
    print('total time cost for walk: ', time_end - time_start)
    return


def exec_embedding():
    '''
    Learn embeddings by optimizing the Skip-gram objective using SGD.
    '''
    logging.info("Initializing creation of the representations...")
    walks = LineSentence('walk_result.txt')
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(get_sim_path() + args.output)
    logging.info("Representations for all disease nodes created.")

    return


def exec_sim_search(args, query_mim_list, k):
    '''
        searching top-k similar diseases for query disease
    '''
    logging.info("searching top-k similar diseases...")

    embedFile = args.output  # embedding file
    sim_d_list = top_k_sim_search(embedFile, query_mim_list, k)

    logging.info("Similarity search for top-k diseases finished.")

    return


def main(args, filelist, query_mim_list, k):
    """
    Main Steps of MultiSimpSim::
        1. calculate the similarity(structural and semantic) between pairwise nodes for each data set
        2. build a multiplex similarity network by associating every counterparts(same nodes) in each network
            --> each network is a similarity network, with each edge referring to the similarity between two nodes
        3. conduct the biased random walk for every node for several times, to generate independent sequences as its context
        4. use the word2vec package to learning the embeddings for every node by its context.
    """
    sim_file_list = exec_SimpSim(args, filelist)        # 2. 对所有数据集计算得到相似关系，并保存输出计算结果

    # sim_file_list = ['combined_file_name1']
    # sim_file_list = ['d_chemical_888d_other_structsim', 'd_genes_888d_other_structsim']
    exec_construct_multi_sim_network(sim_file_list)  # 3. 对上一步计算到的多个相似网络进行关联，得到多源相似网络，在网络上执行随机游走

    exec_embedding()  # 4. 调用skip-gram模型，学习节点的embedding向量

    exec_sim_search(args, query_mim_list, k)


if __name__ == "__main__":
    args = parse_args()

    # filelist = ['mim_cui', 'mim_locus', 'mim_protein']  # 1. 输入数据集, 表型, 染色体, 蛋白质
    # filelist = ['mim_geneid', 'mim_protein']  # 数据集
    # filelist = ['d_chemical_888', 'd_genes_888']
    filelist = ['d_genes_888']

    # query_mim = 'D10652'  # query disease
    query_mim_list = [
        'D10652', 'D11335', 'D11476', 'D12176', 'D12336', 'D12365', 'D12858', 'D12930', 'D13809', 'D1826', 'D1936', 'D2349', 'D2355', 'D2377', 'D2841', 'D3083',
        'D3312', 'D5844', 'D6132', 'D615', 'D7148', 'D83', 'D8469', 'D848', 'D9351', 'D9455', 'D9588'
    ]
    k = 50  # top k

    main(args, filelist, query_mim_list, k)