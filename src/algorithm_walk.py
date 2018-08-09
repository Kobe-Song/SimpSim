
from collections import deque
import numpy as np
import math, random, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
from time import time

from utils import *


def generate_similarity_network(file_list, workers):
    t0 = time.time()
    logging.info('Creating similarity network...')

    # 生成每层结点的相似度网络, 每层单独存储, 并构建多层网络
    os.system("rm " + get_sim_path() + "/save/weights_similarity-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/graphs-layer-*.pkl")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part1, file_list)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time.time()
    os.system("rm " + get_sim_path() + "/save/similarity_nets_weights-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/alias_method_j-layer-*.pkl")
    os.system("rm " + get_sim_path() + "/save/alias_method_q-layer-*.pkl")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part3)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 3: {}s'.format(t))

    # 合并多层网络
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part4)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part5)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_similarity_network_part6)
        job.result()
    t1 = time.time()
    t = t1 - t0
    logging.info('- Time - part 6: {}s'.format(t))

    return


def generate_similarity_network_part1(file_list):
    '''按层存储每个结点间相似度'''
    # parts = workers
    weights_similarity = {}
    graphs = {}

    for layer in range(0, len(file_list)):
        similarity = load_from_disk(file_list[layer])  # 根据文件名读取存储的相似度文件

        # 初始化该层相似度存储字典
        if (layer not in weights_similarity):
            weights_similarity[layer] = {}
        # 初始化每一层的图
        if (layer not in graphs):
            graphs[layer] = {}

        for vertices, value in similarity.items():
            vx = vertices[0]
            vy = vertices[1]
            weights_similarity[layer][vx, vy] = value       # 存储相似度
            if (vx not in graphs[layer]):
                graphs[layer][vx] = []
            if (vy not in graphs[layer]):
                graphs[layer][vy] = []
            graphs[layer][vx].append(vy)        # 将连接的结点添加进图
            graphs[layer][vy].append(vx)

        logging.info('Layer {} executed.'.format(layer))

    for layer, values in weights_similarity.items():
        save_on_disk(values, 'weights_similarity-layer-' + str(layer))
    for layer, values in graphs.items():
        save_on_disk(values, 'graphs-layer-' + str(layer))
    return


def generate_similarity_network_part3():
    '''同层之间权重赋值'''
    layer = 0
    while (is_pkl('graphs-layer-' + str(layer))):
        graphs = load_from_disk('graphs-layer-' + str(layer))
        weights_similarity = load_from_disk('weights_similarity-layer-' + str(layer))

        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v, neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0

            # 同层边赋予权重
            for n in neighbors:
                if (v, n) in weights_similarity:
                    w = weights_similarity[v, n]
                else:
                    w = weights_similarity[n, v]
                # w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            avg_w = sum_w / len(e_list)

            i = 0
            j = 0
            length = len(e_list)
            while i < length:
                if e_list[j] < avg_w:
                    e_list.remove(e_list[j])
                    graphs[v].pop(j)
                    j -= 1
                else:
                    e_list[j] = e_list[j] / sum_w
                i += 1
                j += 1

            if len(e_list) != len(graphs[v]):
                print(False)
            # e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        os.system("rm " + get_sim_path() + "/save/graphs-layer-" + str(layer) + ".pkl")
        save_on_disk(graphs, 'graphs-layer-' + str(layer))

        save_on_disk(weights, 'similarity_nets_weights-layer-' + str(layer))
        save_on_disk(alias_method_j, 'alias_method_j-layer-' + str(layer))
        save_on_disk(alias_method_q, 'alias_method_q-layer-' + str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_similarity_network_part4():
    '''将多层网络存储到一起'''
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while (is_pkl('graphs-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = load_from_disk('graphs-layer-' + str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving similarityNets on disk...")
    save_on_disk(graphs_c, 'similarity_nets_graphs')
    logging.info('Graphs consolidated.')
    return


def generate_similarity_network_part5():
    alias_method_j_c = {}
    layer = 0
    while (is_pkl('alias_method_j-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = load_from_disk('alias_method_j-layer-' + str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    save_on_disk(alias_method_j_c, 'nets_weights_alias_method_j')

    return


def generate_similarity_network_part6():
    alias_method_q_c = {}
    layer = 0
    while (is_pkl('alias_method_q-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        alias_method_q = load_from_disk('alias_method_q-layer-' + str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    save_on_disk(alias_method_q_c, 'nets_weights_alias_method_q')

    return


def generate_parameters_random_walk():

    logging.info('Loading similarity_nets from disk...')

    sum_weights = {}
    amount_edges = {}

    layer = 0
    while (is_pkl('similarity_nets_weights-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        weights = load_from_disk('similarity_nets_weights-layer-' + str(layer))

        for k, list_weights in weights.items():
            if (layer not in sum_weights):
                sum_weights[layer] = 0
            if (layer not in amount_edges):
                amount_edges[layer] = 0

            for w in list_weights:
                sum_weights[layer] += w
                amount_edges[layer] += 1

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    average_weight = {}
    for layer in list(sum_weights.keys()):
        average_weight[layer] = sum_weights[layer] / amount_edges[layer]

    logging.info("Saving average_weights on disk...")
    save_on_disk(average_weight, 'average_weight')

    amount_neighbours = {}

    layer = 0
    while (is_pkl('similarity_nets_weights-layer-' + str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        weights = load_from_disk('similarity_nets_weights-layer-' + str(layer))

        amount_neighbours[layer] = {}

        for k, list_weights in weights.items():
            cont_neighbours = 0
            for w in list_weights:
                if (w > average_weight[layer]):
                    cont_neighbours += 1
            amount_neighbours[layer][k] = cont_neighbours

        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving amount_neighbours on disk...")
    save_on_disk(amount_neighbours, 'amount_neighbours')


def generate_random_walks_large_graphs(num_walks, walk_length, workers, vertices):

    logging.info('Loading similarity_nets from disk...')

    graphs = load_from_disk('similarity_nets_graphs')
    alias_method_j = load_from_disk('nets_weights_alias_method_j')
    alias_method_q = load_from_disk('nets_weights_alias_method_q')
    amount_neighbours = load_from_disk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time.time()

    walks = deque()
    initialLayer = 0

    parts = workers

    with ProcessPoolExecutor(max_workers = workers) as executor:

        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            logging.info("Execution iteration {} ...".format(walk_iter))
            # walk = exec_ramdom_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours)
            walk = exec_ramdom_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length)
            walks.extend(walk)
            logging.info("Iteration {} executed.".format(walk_iter))

    t1 = time.time()
    logging.info('RWs created. Time : {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)


def generate_random_walks(num_walks, walk_length, workers, vertices):

    logging.info('Loading similarity_nets on disk...')

    graphs = load_from_disk('similarity_nets_graphs')
    alias_method_j = load_from_disk('nets_weights_alias_method_j')
    alias_method_q = load_from_disk('nets_weights_alias_method_q')
    # amount_neighbours = load_from_disk('amount_neighbours')

    logging.info('Creating RWs...')
    t0 = time.time()

    walks = deque()
    initialLayer = 0

    if (workers > num_walks):
        workers = num_walks

    with ProcessPoolExecutor(max_workers = workers) as executor:
        futures = {}
        for walk_iter in range(num_walks):
            random.shuffle(vertices)
            # job = executor.submit(exec_ramdom_walks_for_chunck, vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours)
            job = executor.submit(exec_ramdom_walks_for_chunck, vertices, graphs, alias_method_j, alias_method_q, walk_length)
            futures[job] = walk_iter
            #part += 1
        logging.info("Receiving results...")
        for job in as_completed(futures):
            walk = job.result()
            r = futures[job]
            logging.info("Iteration {} executed.".format(r))
            walks.extend(walk)
            del futures[job]

    t1 = time.time()
    logging.info('RWs created. Time: {}m'.format((t1 - t0) / 60))
    logging.info("Saving Random Walks on disk...")
    save_random_walks(walks)

# def exec_ramdom_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length, amount_neighbours):
def exec_ramdom_walks_for_chunck(vertices, graphs, alias_method_j, alias_method_q, walk_length):
    '''将vertices中每个结点作为起始点, 进行随机游走'''
    walks = deque()
    for v in vertices:
        # walks.append(exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours))
        walks.append(exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length))
    return walks

# def exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length, amount_neighbours):
def exec_random_walk(graphs, alias_method_j, alias_method_q, v, walk_length):
    original_v = v
    t0 = time.time()
    initialLayer = 0
    layer = initialLayer

    path = deque()
    path.append(v)

    num_graphs = len(graphs)
    prob_move = 1 / num_graphs

    while len(path) < walk_length:
        r = random.random()

        current_layer = layer

        for l in range(num_graphs):
            if r < (l + 1) * prob_move:
                # 跳转到l层
                layer = l
                # 如果仍在当前层, 则添加新结点
                if current_layer == layer:
                    v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
                    path.append(v)
                break


        # 停留在当前层
        # if (r < 0.5):
        #     v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
        #     path.append(v)

        # else:
        #     r = random.random()
        #     limiar_moveup = prob_moveup(amount_neighbours[layer][v])
        #     # 返回上一层
        #     if (r > limiar_moveup):
        #         if (layer > initialLayer):
        #             layer = layer - 1
        #     # 去下一层
        #     else:
        #         if ((layer + 1) in graphs and v in graphs[layer + 1]):
        #             layer = layer + 1

    t1 = time.time()
    logging.info('RW - vertex {}. Time : {}s'.format(original_v, (t1 - t0)))

    return path


def chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer):
    v_list = graphs[layer][v]

    idx = alias_draw(alias_method_j[layer][v], alias_method_q[layer][v])
    v = v_list[idx]

    return v


def prob_moveup(amount_neighbours):
    '''返回游走到k+1层的概率'''
    x = math.log(amount_neighbours + math.e)
    p = (x / (x + 1))
    return p


def save_random_walks(walks):
    with open('walk_result.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v) + ' '
            line += '\n'
            file.write(line)
    return