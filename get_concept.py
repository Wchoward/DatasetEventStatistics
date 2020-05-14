import xml.etree.ElementTree as ET
import ast
import pandas as pd
from langconv import *
import jieba.posseg as posseg


def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line


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
    return lst


if __name__ == '__main__':
    lst = get_P('data/ImplicitECD.Tuple.forTag.xml')
    # lst = get_P('data/test.xml')
    segmentation_P(lst)
