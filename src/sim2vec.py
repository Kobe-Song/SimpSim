import numpy as np
import random, sys, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time

from utils import *
from algorithm_walk import *
import graph


def construct_multi_sim_network(file_list, workers=4):
    '''构建多层带权完全图'''
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network, file_list, workers)

        job.result()

    return


def preprocess_parameters_random_walk():
    '''初始化随机游走的参数'''
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_parameters_random_walk)

        job.result()

    return


def simulate_walks(num_walks, walk_length, workers = 4):
    '''随机游走'''

    vertices = load_from_disk('vertices')
    # for large graphs, it is serially executed, because of memory use.
    if (len(vertices) > 500000):

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_random_walks_large_graphs, num_walks, walk_length, workers, vertices)

            job.result()

    else:

        with ProcessPoolExecutor(max_workers=1) as executor:
            job = executor.submit(generate_random_walks, num_walks, walk_length, workers, vertices)

            job.result()

    return
