# -*- coding:utf-8 -*-
import fm
import os
import xml.etree.ElementTree as ET
import pandas as pd


def process_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lst = []
    for sample in root:
        count = 0
        for event in sample[1].findall('event'):
            if (event.get('cause') == 'Y'):
                count = count + 1
        lst.append([int(sample.get('id')), count, 0])
    df = pd.DataFrame(lst, columns=['sample_id', 'verb_event_num', 'none_event_num'])
    df.to_csv("data/result.csv", index=False)


if __name__ == '__main__':
    process_file('data/ImplicitECD.Tuple.forTag.xml')
