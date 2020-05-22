import xml.etree.ElementTree as ET
import numpy as np
import ast
import fm
import pandas as pd
from langconv import *
import re
import time
import json


# from bert_serving.client import BertClient
#
# bc = BertClient()


# 简体转换为繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


# 繁体转换为简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


# 将xml转化为输入bert的语料list
def generate_corpus_list(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lst = []
    emotion_lst = []
    for sample in root:
        emotion_lst.append(sample.get('emotion_category'))
        sample_text = [chs_to_cht(sample[0].text)]
        for event in sample[1]:
            event_content = ast.literal_eval(event.text)
            event_text = event_content['Att_Subj'] + event_content['Subj'] + event_content['Adv'] + event_content['P'] + \
                         event_content['Cpl'] + event_content['Att_Obj'] + event_content['Obj']
            sample_text.append(chs_to_cht(event_text))
        lst.append(sample_text)
    np.save('data/emotion.npy', np.array(emotion_lst))
    return lst


def generate_emotion_cause_npz(file_path, lst):
    L = [len(x) for x in lst]
    maxLen = max(L)
    tree = ET.parse(file_path)
    root = tree.getroot()
    emotion_cause_lst = []
    for sample in root:
        emotion_cause = [0]
        for event in sample[1]:
            if event.get('cause') == 'N':
                emotion_cause.append(0)
            else:
                emotion_cause.append(1)
        arr = np.array(emotion_cause)
        arr = np.pad(arr, (0, maxLen - len(arr)), 'constant')
        emotion_cause_lst.append(arr)
    res = np.concatenate(emotion_cause_lst)
    np.save('data/emotion_cause.npy', res)
    return res


def get_embedding(lst):
    L = [len(x) for x in lst]
    maxLen = max(L)
    tmplst = []
    for sample in lst:
        tmp = bc.encode(sample)
        tmp = np.pad(tmp, ((0, maxLen - len(tmp)), (0, 0)), 'constant')
        tmplst.append(tmp[np.newaxis, :])
    res = np.concatenate(tmplst)
    return res


if __name__ == '__main__':
    raw_lst = generate_corpus_list('data/ImplicitECD.Tuple.forTag.xml')
    generate_emotion_cause_npz('data/ImplicitECD.Tuple.forTag.xml', raw_lst)
    # embedding = get_embedding(raw_lst)
    # np.save('origin_embedding.npy', embedding)
