import xml.etree.ElementTree as ET
import ast
import pandas as pd
from langconv import *
import jieba.posseg as posseg
import requests
from requests.adapters import HTTPAdapter
import re
import json

re_concept = re.compile(
    r'/a/\[/r/(?P<Relation>(Causes|MotivatedByGoal|CausesDesire))/,/c/zh/(?P<Concept1>.*?)/,/c/zh/(?P<Concept2>.*?)/\]')
request = requests.Session()
request.mount('http://', HTTPAdapter(max_retries=3))
request.mount('https://', HTTPAdapter(max_retries=3))


def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


def get_concept_triplet_list(word):
    obj = request.get('http://api.conceptnet.io/c/zh/' + word, timeout=5).json()
    lst = []
    for edge in obj['edges']:
        s = edge['@id']
        match = re_concept.match(s)
        if match:
            dict = match.groupdict()
            weight = edge['weight']
            dict['Weight'] = weight
            lst.append(dict)
    return lst


def generate_conceptNet_json(file_path):
    conceptNet_dict = {}
    lst = get_P(file_path)
    (lst, word_list) = segmentation_P(lst)
    for word in word_list:
        conceptNet_dict[word] = get_concept_triplet_list(word)
    with open('data/conceptDict.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(conceptNet_dict, ensure_ascii=False))


def generate_ConceptNet_csv(file_path):
    df = pd.read_csv(file_path)
    for i in range(df.shape[0]):
        word = df.loc[i, 'words']
        lst = get_concept_triplet_list(word)
        df.loc[i, 'conceptNet'] = lst
        df.to_csv(file_path, index=False)


def generate_all_event_P_ConceptNet_csv(file_path, dict_path):
    all_event_P = pd.read_csv(file_path)
    conceptNet_dict = pd.read_csv(dict_path)
    for i in range(all_event_P.shape[0]):
        str_words = all_event_P.loc[i, 'segmentation words']
        lst_words = ast.literal_eval(str_words)
        conceptNet_list = []
        for word in lst_words:
            word_conceptNet = conceptNet_dict.loc[conceptNet_dict['words'] == word, 'conceptNet']
            word_conceptNet_lst = ast.literal_eval(word_conceptNet.values.tolist()[0])
            conceptNet_list.append(word_conceptNet_lst)
        all_event_P.loc[i, 'conceptNet'] = json.dumps(conceptNet_list, ensure_ascii=False)
        all_event_P.to_csv(file_path, index=False)


# 输入xml文件获取每个语料每个事件的P部分，返回list
def get_P(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lst = []
    for sample in root:
        for event in sample[1]:
            sample_id = int(sample.get('id'))
            event_id = int(event.get('id'))
            event_content = ast.literal_eval(event.text)
            event_P = chs_to_cht(event_content['P'])
            lst.append([sample_id, event_id, event_P])
    # df = pd.DataFrame(lst, columns=['sample_id', 'event_id', 'event_P'])
    # df.to_csv("data/all_event_P.csv", index=False)
    return lst


# get_P后对 list进行 P部分的分词
def segmentation_P(lst):
    word_list = []
    for i in lst:
        seg = posseg.lcut(i[2])
        word = []
        for pair in seg:
            if pair.flag == 'v' or pair.flag == 'a' or pair.flag == 'n':
                word.append(pair.word)
                word_list.append(pair.word)
        if len(word) == 0:
            word.append(i[2])
            word_list.append(i[2])
        i.append(word)
    df = pd.DataFrame(lst, columns=['sample_id', 'event_id', 'event_P', 'segmentation words'])
    df.to_csv("data/all_event_P.csv", index=False)
    # word_list = list(set(word_list))
    # df2 = pd.DataFrame(word_list, columns=['words'])
    # df2.to_csv("data/conceptList.csv", index=False)
    return lst, word_list


if __name__ == '__main__':
    # lst = get_P('data/ImplicitECD.Tuple.forTag.xml')
    # lst = get_P('data/test.xml')
    # segmentation_P(lst)
    # generate_ConceptNet_csv('data/conceptList.csv')
    # generate_all_event_P_ConceptNet_csv('data/all_event_P.csv', 'data/conceptList.csv')
    generate_conceptNet_json('data/test.xml')
