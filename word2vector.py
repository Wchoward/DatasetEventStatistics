import jieba
import jieba.analyse
import codecs
import sys
import re
import os
from gensim import utils
import gensim.models
import gensim
import fm
from langconv import *
import logging
from gensim.models import Word2Vec
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


class MyCorpus:
    def __init__(self, seg_root):
        self.seg_root = seg_root

    def __iter__(self):
        for filename in os.listdir(self.seg_root):
            filepath = os.path.join(self.seg_root, filename)
            for line in codecs.open(filepath, 'r', encoding='utf-8'):
                yield utils.simple_preprocess(line)


# def get_word2vec_embedding(lst):
#     seg_lst = []
#     for sentence in lst:
#         seg_lst.append(sentence.split())


if __name__ == '__main__':
    # corpus_segment('data/zh_wiki', 'data/seg_wiki')
    # sentences = MyCorpus('data/seg_wiki')
    # 训练word2vec
    model = gensim.models.Word2Vec(LineSentence('data/merge_seg_wiki.txt'), min_count=5, size=300, workers=4, sg=0)

    # 保存模型
    model.save('model/zh_wiki.model')
