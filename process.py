# -*- coding:utf-8 -*-
import fm
import os
import xml.etree.ElementTree as ET
import pandas as pd


def process_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lst = []
    # sample1 = root[0]
    # events1 = root[0][1]
    # event1 = root[0][1][0]
    for sample in root:
        count = 0
        for event in sample[1].findall('event'):
            if event.get('cause') == 'Y':
                count = count + 1
            else:
                sample[1].remove(event)
        lst.append([int(sample.get('id')), count, 0])
    # 生成删除cause为N节点的xml
    tree.write("data/data_without_Ncause.xml", encoding="utf-8", xml_declaration=True)
    df = pd.DataFrame(lst, columns=['sample_id', 'verb_event_num', 'noun_event_num'])
    # 生成原因整体数量统计表（全部当作动词事件）
    df.to_csv("data/raw_cause_number_result.csv", index=False)


def process_noun_event_table(dataset_path):
    noun_table = pd.read_csv('data/noun_table.csv')
    result_table = pd.read_csv('data/raw_cause_number_result.csv')
    tree = ET.parse(dataset_path)
    root = tree.getroot()
    for noun_row in noun_table.itertuples():
        sample_id = getattr(noun_row, 'sample_id')
        event_id = getattr(noun_row, 'event_id')
        if root[sample_id - 1][1][event_id - 1].get('cause') == 'Y':
            result_table.loc[sample_id - 1, 'noun_event_num'] = result_table.loc[sample_id - 1, 'noun_event_num'] + 1
    # 生成抽取出名词事件的数量统计表,后续需要减一下
    result_table.to_csv('data/cause_number_result.csv', index=False)


if __name__ == '__main__':
    process_file('data/ImplicitECD.Tuple.forTag.xml')
    process_noun_event_table('data/ImplicitECD.Tuple.forTag.xml')
