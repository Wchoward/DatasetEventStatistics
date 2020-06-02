import jieba
import jieba.analyse
import codecs
import os
import re
import numpy as np
import pandas as pd
import gensim.models
import gensim
import fm
from langconv import *
import logging
from process_NRC_VAD import get_emotion_intensity
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

wikiDocBegin = re.compile(r'^<doc.*>$')
wikiDocEnd = re.compile(r'^</doc>$')


# model = Word2Vec.load('model/zh_wiki.model')


def segment(ori_file, seg_file, wiki=True):
    """
    对文件进行分词
    Args:
        ori_file: 原始文本文件
        seg_file: 分词后的文本文件
        wiki: 是否为wiki数据,如果是wiki，则去掉包含<doc></doc>的行
    Returns:
        None
    """
    with codecs.open(seg_file, 'w+', encoding='utf-8') as fw:
        for line in codecs.open(ori_file, 'r', encoding='utf-8'):
            if line == '\n':
                continue

            if wiki and (wikiDocBegin.match(line) or wikiDocEnd.match(line)):
                line_seg = '\n'
            else:
                # 这里可以过滤停用词
                # for word in jieba.cut(line):
                #     if word in stopwords:
                #         ...
                line_seg = ' '.join(jieba.cut(line))
            fw.write(line_seg)


def corpus_segment(ori_root, seg_root):
    """
    将一个目录下的文本文件进行分词，并将分词结果存储到目标目录
    Args:
        ori_root: 原始文件目录
        seg_root: 分词结果文件目录
    Returns:
        None
    """
    if os.path.exists(ori_root) and os.path.exists(seg_root):
        for filename in os.listdir(ori_root):
            filepath = os.path.join(ori_root, filename)
            savepath = os.path.join(seg_root, filename)
            print('%s -> %s' % (filepath, savepath))
            segment(filepath, savepath)
    else:
        print("%s or %s is not exists!" % (ori_root, seg_root))


def get_all_origin_seg_text(csv_dir_path):
    """
    将我们语料中的原始文本分词后合并入一个文件 用于后续训练word2vec
    :param csv_dir_path: 语料csv目录
    :return: 保存为一个全部的txt
    """
    file_path_lst = []
    all_origin_seg_lst = []
    file_path_lst = fm.get_filelist(csv_dir_path, file_path_lst)
    for file_path in file_path_lst:
        df = pd.read_csv(file_path)
        column0 = df.iloc[:, 0].values.tolist()
        uni_col = list(set(column0))
        all_origin_seg_lst += uni_col
    fm.save_file('data/seg_corpus/all_origin_seg.txt', all_origin_seg_lst)


def merge_dir_texts(dir_path, result_path):
    """
    合并文件夹中的内容到指定的txt文件内
    :param dir_path: 需要合并的文件夹
    :param result_path: 结果保存的文件目录
    :return:
    """
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(dir_path)
    # 打开当前目录下的result.txt文件，如果没有则创建
    file = open(result_path, 'w+', encoding="utf-8")
    # 向文件中写入字符
    # 先遍历文件名
    for filename in filenames:
        filepath = dir_path + '/'
        filepath = filepath + filename
        # 遍历单个文件，读取行数
        for line in open(filepath, encoding="utf-8"):
            file.writelines(line)
        file.write('\n')
    # 关闭文件
    file.close()


def load_word2vec_model_text_to_dict(file_path, ):
    """
    将word2vector模型生成的txt文件转化为dict
    :param file_path:
    :return:
    """
    dict_model = {}
    with open(file_path, encoding='utf-8') as f_origin:
        next(f_origin)
        for line in f_origin:
            tmp = line.strip(' \n').split(' ', 1)
            k = tmp[0]
            v = tmp[1]
            dict_model[k] = v
    return dict_model


def save_dict_to_word2vec_model_text(model_dict, file_path):
    print('length:', len(model_dict))
    words = list(model_dict.keys())
    vectors = list(model_dict.values())
    lst = [str(len(model_dict)) + ' 300']
    for i in range(len(model_dict)):
        lst.append(words[i] + ' ' + vectors[i])
    fm.save_file(file_path, lst)


def merge_word2vec_model_texts(origin_w2v_path, concept_w2v_path, merge_w2v_path):
    """
    合并两个text的word2vector模型为一个
    :param origin_w2v_path:
    :param concept_w2v_path:
    :param merge_w2v_path:
    :return:
    """
    # load origin model text
    dict_origin = load_word2vec_model_text_to_dict(origin_w2v_path)
    # load conceptNet model text
    dict_concept = load_word2vec_model_text_to_dict(concept_w2v_path)
    # merge dict
    dict_merge = dict(dict_origin, **dict_concept)
    print('length:', len(dict_merge))
    words = list(dict_merge.keys())
    vectors = list(dict_merge.values())
    lst = [str(len(dict_merge)) + ' 300']
    for i in range(len(dict_merge)):
        lst.append(words[i] + ' ' + vectors[i])
    fm.save_file(merge_w2v_path, lst)


def get_sent_embedding(lst, max_len):
    """
    获取列表的embedding矩阵
    :param lst: 输入的csv中的某一列转化后的文本的列表
    :param max_len: 最大分词数量，来padding
    :return: 矩阵
    """
    seg_lst = []
    for sentence in lst:
        seg_lst.append(str(sentence).split())
    # L = [len(x) for x in seg_lst]
    # max_len = max(L)
    print('max_len:', max_len)
    sent_embedding_lst = []
    for sentence in seg_lst:
        word_embedding_lst = []
        for word in sentence:
            # if word not in model:
            #     print(word)
            word_embedding_lst.append(model[word][np.newaxis, :])
        sent_embedding = np.concatenate(word_embedding_lst)
        sent_embedding_lst.append(
            np.pad(sent_embedding, ((0, max_len - len(sent_embedding)), (0, 0)), 'constant')[np.newaxis, :])
    embedding = np.concatenate(sent_embedding_lst)
    return embedding


def get_sent_embedding_integrated_conceptnet(sent_lst, max_len, concept_dict, lam):
    seg_lst = []
    for sentence in sent_lst:
        seg_lst.append(str(sentence).split())
    # L = [len(x) for x in seg_lst]
    # max_len = max(L)
    print('max_len:', max_len)
    sent_embedding_lst = []
    for sentence in seg_lst:
        word_embedding_lst = []
        for word in sentence:
            if word not in concept_dict.keys():
                word_embedding_lst.append(model[word][np.newaxis, :])
            else:
                weight_lst = []
                concept_embedding_lst = []
                for c_k in concept_dict[word]:
                    e_k = get_emotion_intensity(NRC, word, lam)
                    if e_k is not None:
                        if c_k['Concept'] in model_conceptnet:
                            weight_k = e_k * c_k['Weight']
                            weight_lst.append(weight_k)
                            concept_embedding_lst.append(model_conceptnet[c_k['Concept']])
                weight_lst = np.array(weight_lst)
                if not concept_embedding_lst:
                    integrated_embedding = model[word]
                else:
                    concept_embedding_lst = np.array(concept_embedding_lst)
                    alpha_lst = np.exp(weight_lst) / sum(np.exp(weight_lst))
                    integrated_embedding = (np.dot(alpha_lst, concept_embedding_lst) + model[word]) / 2
                word_embedding_lst.append(integrated_embedding[np.newaxis, :])
        sent_embedding = np.concatenate(word_embedding_lst)
        sent_embedding_lst.append(
            np.pad(sent_embedding, ((0, max_len - len(sent_embedding)), (0, 0)), 'constant')[np.newaxis, :])
    embedding = np.concatenate(sent_embedding_lst)
    return embedding


def generate_embedding_npy(file_path, dir_path, lam):
    """
    将csv文件的三列转化为embedding矩阵，分别保存为npy
    :param lam: lambda
    :param file_path: csv文件目录
    :param dir_path: 保存的路径
    :return:
    """
    df = pd.read_csv(file_path)
    concept_dict = fm.load_dict_json('data/conceptNet/simplified_concept_dict.json')
    column0 = df.iloc[:, 0].values.tolist()
    embedding0 = get_sent_embedding(column0, 169)
    # embedding0 = get_sent_embedding_integrated_conceptnet(column0, 169, concept_dict, lam)
    np.save(dir_path + '/origin_text.npy', embedding0)
    column1 = df.iloc[:, 1].values.tolist()
    embedding1 = get_sent_embedding(column1, 44)
    # embedding1 = get_sent_embedding_integrated_conceptnet(column1, 44, concept_dict, lam)
    np.save(dir_path + '/cause_event.npy', embedding1)
    embedding2 = df.iloc[:, 2].values
    np.save(dir_path + '/if_cause.npy', embedding2)


def generate_file_list_embedding(csv_dir_path, embedding_dir_path, lam):
    """
    将整个文件夹内的所有的csv进行embedding转化，保存到指定的目录下
    :param lam: lambda
    :param csv_dir_path: 输入的csv的目录路径
    :param embedding_dir_path: 输出的文件保存的目录路径
    :return:
    """
    re_emotion_category = re.compile(r'.*?/.*?/ImplicitECD.(.*?).tokenization.csv')
    file_path_lst = []
    file_path_lst = fm.get_filelist(csv_dir_path, file_path_lst)
    for file_path in file_path_lst:
        category = re_emotion_category.match(file_path).group(1)
        os.makedirs(embedding_dir_path + '/' + category, exist_ok=True)
        print(category + ':')
        generate_embedding_npy(file_path, embedding_dir_path + '/' + category, lam)


if __name__ == '__main__':
    # 将我们语料中的原始文本分词后合并入一个文件 用于后续训练word2vec
    # get_all_origin_seg_text('data/emotion_category')
    # corpus_segment('data/zh_wiki', 'data/seg_wiki')
    # merge_dir_texts('data/seg_wiki', 'data/seg_corpus/merge_seg_wiki.txt')
    # merge_dir_texts('data/seg_corpus', 'data/merge_seg_all_corpus.txt')
    # 训练word2vec
    # model = gensim.models.Word2Vec(LineSentence('data/merge_seg_all_corpus.txt'), min_count=1, size=300, workers=4, sg=0)
    # 保存模型
    # model.save('model/origin_w2v_model/zh_wiki.model')
    # merge_word2vec_model_texts('model/origin_word2vec_embedding.txt', 'model/zhs_conceptnet_embedding.txt',
    #                            'model/merge_w2v_embedding.txt')
    NRC = fm.load_dict_json('data/NRC_VAD/NRC.json')
    # 调用模型
    model = Word2Vec.load('model/origin_w2v_model/zh_wiki.model')
    # model_conceptnet = KeyedVectors.load_word2vec_format("model/conceptnet_embedding/filtered_conceptnet_embedding.txt",
    #                                                      binary=False)
    generate_file_list_embedding('data/emotion_category', 'data/origin_w2v_emotion_category', 0)
    # generate_file_list_embedding('data/emotion_category', 'data/w2v_emotion_category_lambda_0', 0)
    # os.system('tar zcvf data/w2v_emotion_category_lambda_0.tar.gz data/w2v_emotion_category_lambda_0')
    # os.system('rm -rf data/w2v_emotion_category_lambda_0')
    # generate_file_list_embedding('data/emotion_category', 'data/w2v_emotion_category_lambda_0.25', 0.25)
    # os.system('tar zcvf data/w2v_emotion_category_lambda_0.25.tar.gz data/w2v_emotion_category_lambda_0.25')
    # os.system('rm -rf data/w2v_emotion_category_lambda_0.25')
    # generate_file_list_embedding('data/emotion_category', 'data/w2v_emotion_category_lambda_0.5', 0.5)
    # os.system('tar zcvf data/w2v_emotion_category_lambda_0.5.tar.gz data/w2v_emotion_category_lambda_0.5')
    # os.system('rm -rf data/w2v_emotion_category_lambda_0.5')
    # generate_file_list_embedding('data/emotion_category', 'data/w2v_emotion_category_lambda_0.75', 0.75)
    # os.system('tar zcvf data/w2v_emotion_category_lambda_0.75.tar.gz data/w2v_emotion_category_lambda_0.75')
    # os.system('rm -rf data/w2v_emotion_category_lambda_0.75')
    # generate_file_list_embedding('data/emotion_category', 'data/w2v_emotion_category_lambda_1', 1)
    # os.system('tar zcvf data/w2v_emotion_category_lambda_1.tar.gz data/w2v_emotion_category_lambda_1')
    # os.system('rm -rf data/w2v_emotion_category_lambda_1')
