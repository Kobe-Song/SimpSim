
# -*- coding: utf-8 -*-
import sys
import argparse, logging
import numpy as np
from time import time

from graph import *
from pathsim import *
from structSim import *
from pathsim import *
import utils


logging.basicConfig(filename='SimpSim.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')


def parse_args():
    '''
    Parses the SimpSim arguments.
    '''
    parser = argparse.ArgumentParser(description="Run SimpSim.")

    # 输入文件（节点、边文件）
    parser.add_argument('--input', nargs='?', default='data/mim_dname_pid_des_cui.xlsx',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='save/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--until-layer', type=int, default=None,
                        help='Calculation until the layer.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    parser.add_argument('--OPT1', default=False, type=bool,
                        help='optimization 1')
    parser.add_argument('--OPT2', default=False, type=bool,
                        help='optimization 2')
    parser.add_argument('--OPT3', default=False, type=bool,
                        help='optimization 3')
    return parser.parse_args()





def exec_SimpSim(args):
    """
        1. read_graph: read file and build graph(HIN, stored by dict)
        2. exec_pathSim: cal meta-path-based similarity for pairwise nodes (represents semantic similarity)
        3. exec_struc2vec: cal structural similarity for pairwise nodes (represents structural similarity)
        4. combine PathSim and StructSim together and get the final similarity
    """
    G = read_graph(args)
    exec_pathSim(G)
    # exec_StructSim(G, args)
    # exec_combine_sim(G, args)


def read_graph(args):
    '''
    Reads file and builds a HIN network.
    '''
    logging.info(" - Loading edge file and build HIN...")

    input_filename = args.input
    G = load_edge_file(input_filename)  # 读取边数据，构建图G，是一个dict(list)类型的字典

    logging.info(" - HIN loaded.")

    print("G length:", len(G))

    return G



def exec_pathSim(G):
    """
    :param G: original d_p dict
    :return: dict with pathsim score
    """
    logging.info(" - Processing PathSim calculation...")

    time_start = time.time()

    cal_pathSim(G)  # calculates PathSim score for all pairwise nodes

    time_end = time.time()
    print('total time cost for pathSim: ', time_end - time_start)

    logging.info(" - PathSim calculation finished.")

    return



def exec_StructSim(args, query_d):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    G = read_graph(args)
    query_list = []

    if type(query_d) != list:
        query_list.append(query_d)
    else:
        query_list = query_d

    # 获取疾病结点的度序列
    # if(args.OPT1):
    #     structSim.generate_degreeSeq_with_bfs_compact()
    # else:
    generate_degreeSeq_with_bfs(G, 5)
    # generate_degreeSeq_with_bfs_query(G, query_list)

    # 计算结点之间的距离
    # if (args.OPT2):
	# 	# 优化2：对于结构完全不同的两个顶点，不计算他们的结构距离
	# 	G.create_vectors()
	# 	G.calc_distances(compactDegree=args.OPT1)
	# else:
    calc_distances_all_vertices(G)
    # calc_distances_query(G, query_list)

    consolide_distances()
    calc_strucSim('all')

    logging.info(" - StructSim calculation finished.")

    return


def exec_combine_sim():
    """
    combine pathsim and structsim together and get the final similarity value for each pairwise nodes.
    here we use inequality equation to combine them
    """



    return



def main(args):
    """
    1. cal distance-->similarity between pair nodes
    2. build multi-layer network
        --> each layer is a complete graph, refering to a k-neighborhood
        --> each edge in each layer indicates the structure simlarity between pair nodes
    """
    # exec_SimpSim(args)
    exec_StructSim(args, 242900)



if __name__ == "__main__":
    args = parse_args()
    main(args)

