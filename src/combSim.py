from utils import *


def combine_sim(pathSim_file_name, structSim_file_name, index):

    # combine_sim()

    d_pair_pathsim_dict = load_from_disk(pathSim_file_name)  # 读取根据疾病元路径计算出来的疾病相似度（字典）
    print("d_pair_pathsim_dict length: ", len(d_pair_pathsim_dict))

    d_pair_structsim_dict = load_from_disk(structSim_file_name)  # 读取根据疾病网络结构计算出来的疾病相似度（字典）
    print("d_pair_structsim_dict length: ", len(d_pair_structsim_dict))

    d_pair_sim_combine_dict = {}

    # 读取语义相似度文件和结构相似度文件

    for key in list(d_pair_structsim_dict.keys()):  # 从结构相似字典依次取出一个记录(这个字典应该是存储了任意两点之间的结构)
        pathsim_score = 0  # 语义相似度初始化为0

        if key in list(d_pair_pathsim_dict.keys()):  # 字典只保存了相似度值大于0的记录
            pathsim_score = d_pair_pathsim_dict[key]  # 语义相似度值>0，获取该相似度值

        d_pair_sim_combine_dict[key] = cal_mean_sim(d_pair_structsim_dict[key], pathsim_score)  # 使用算术平均式计算合并之后的相似度

    combined_file_name = 'combined_file_name' + str(index)
    save_on_disk(d_pair_sim_combine_dict, combined_file_name)  # 保存最终相似度计算结果

    return combined_file_name


def cal_mean_sim(score1, score2):
    return (score1 + score2) / 2.0

def cal_harmonic_sim(score1, score2):
    return 2.0 / (1 / score1 + 1 / score2)