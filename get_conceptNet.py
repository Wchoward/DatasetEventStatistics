import xml.etree.ElementTree as ET
import ast
import fm
import pandas as pd
from langconv import *
import jieba.posseg as posseg
import requests
from requests.adapters import HTTPAdapter
import re
import time
import json

re_concept = re.compile(
    r'/a/\[/r/(?P<Relation>(Causes|MotivatedByGoal|CausesDesire))/,/c/zh/(?P<Concept1>.*?)/,/c/zh/(?P<Concept2>.*?)/\]')
request = requests.Session()
request.mount('http://', HTTPAdapter(max_retries=3))
request.mount('https://', HTTPAdapter(max_retries=3))


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


# 调用conceptNet api 获取对应单词的三元组
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


# 生成ConceptNet字典，输出json
def generate_conceptNet_json(file_path):
    conceptNet_dict = {}
    lst = get_adv_p_cpl(file_path)
    (lst, word_list) = segmentation(lst)
    for word in word_list:
        conceptNet_dict[word] = get_concept_triplet_list(word)
    conceptNet_dict[''] = []
    with open('data/conceptNet/conceptDict.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(conceptNet_dict, ensure_ascii=False))


# 抽取出ConceptNetDict中的空value，形成NullConceptDict
def generate_null_conceptNet_json(file_path_origin):
    null_conceptNet_dict = {}
    with open(file_path_origin, encoding='utf-8') as f:
        conceptNet_dict = json.load(f)
    for k, v in conceptNet_dict.items():
        if not v:
            null_conceptNet_dict[k] = v
    with open('data/conceptNet/nullConceptDict.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(null_conceptNet_dict, ensure_ascii=False))


import synonyms


# 使用近义词表扩充conceptDict.json
def expand_conceptNet_json():
    with open('data/conceptNet/expand_conceptDict.json', encoding='utf-8') as f:
        conceptNet_dict = json.load(f)
    with open('data/conceptNet/nullConceptDict.json', encoding='utf-8') as f:
        nullConcept_dict = json.load(f)
    nullConcept_dict_keys = list(nullConcept_dict.keys())
    sub_dict = dict([(key, conceptNet_dict[key]) for key in nullConcept_dict_keys[2006:]])

    for key, value in sub_dict.items():
        if not value:
            word = cht_to_chs(key)
            synonym_words = synonyms.nearby(word)[0]
            for synonym_word in synonym_words:
                try:
                    # time.sleep(1)
                    tmp = get_concept_triplet_list(chs_to_cht(synonym_word))
                    if tmp:
                        conceptNet_dict[key] = tmp
                        break
                except Exception:
                    print(key)
                    with open('data/conceptNet/expand_conceptDict.json', 'w', encoding='utf-8') as f:
                        f.write(json.dumps(conceptNet_dict, ensure_ascii=False))
    with open('data/conceptNet/expand_conceptDict.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(conceptNet_dict, ensure_ascii=False))


# 输入xml文件获取每个语料每个事件的Adv,P,Cpl部分，返回list
def get_adv_p_cpl(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lst = []
    for sample in root:
        for event in sample[1]:
            sample_id = int(sample.get('id'))
            event_id = int(event.get('id'))
            event_content = ast.literal_eval(event.text)
            event_Adv = chs_to_cht(event_content['Adv'])
            event_P = chs_to_cht(event_content['P'])
            event_Cpl = chs_to_cht(event_content['Cpl'])
            lst.append([sample_id, event_id, event_Adv, event_P, event_Cpl])
    # df = pd.DataFrame(lst, columns=['sample_id', 'event_id', 'event_P'])
    # df.to_csv("data/all_event_P.csv", index=False)
    return lst


# 对获取到的短语进行jieba分词
def segmentation(lst):
    word_list = []
    for event in lst:
        word = []
        for i in range(2, 5):
            seg = posseg.lcut(event[i])
            for pair in seg:
                if pair.flag == 'v' or pair.flag == 'a' or pair.flag == 'n':
                    word.append(pair.word)
                    word_list.append(pair.word)
        if len(word) == 0:
            word.append(event[3])
            word_list.append(event[3])
        word = list(set(word))
        event.append(word)
    df = pd.DataFrame(lst, columns=['sample_id', 'event_id', 'event_Adv', 'event_P', 'event_Cpl', 'segmentation words'])
    df.to_csv("data/all_event_Adv_P_Cpl.csv", index=False)
    word_list = list(set(word_list))
    # df2 = pd.DataFrame(word_list, columns=['words'])
    # df2.to_csv("data/conceptList.csv", index=False)
    return lst, word_list


# 生成所有事件的Adv,P,Cpl的关键词的ConceptNet
def generate_all_event_Adv_P_Cpl_ConceptNet_csv(file_path, dict_path):
    all_event_Adv_P_Cpl = pd.read_csv(file_path)
    with open(dict_path, encoding='utf-8') as f:
        conceptNet_dict = json.load(f)
    for i in range(all_event_Adv_P_Cpl.shape[0]):
        str_words = all_event_Adv_P_Cpl.loc[i, 'segmentation words']
        lst_words = ast.literal_eval(str_words)
        conceptNet_list = []
        for word in lst_words:
            word_conceptNet = conceptNet_dict[word]
            conceptNet_list.append(word_conceptNet)
        all_event_Adv_P_Cpl.loc[i, 'conceptNet'] = json.dumps(conceptNet_list, ensure_ascii=False)
    all_event_Adv_P_Cpl.to_csv(file_path, index=False)


if __name__ == '__main__':
    # generate_conceptNet_json('data/ImplicitECD.Tuple.forTag.xml')
    # generate_all_event_Adv_P_Cpl_ConceptNet_csv('data/all_event_Adv_P_Cpl.csv', 'data/conceptDict.json')
    generate_null_conceptNet_json('data/conceptNet/expand_conceptDict.json')
    # expand_conceptNet_json()
