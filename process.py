# -*- coding:utf-8 -*-
import fm
import os
import xml.etree.ElementTree as ET


def process_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    for sample in root:
        count = 0
        for event in sample[1].findall('event'):
            if (event.get('cause') == 'Y'):
                count = count + 1
        print(sample.get('id'), count)


if __name__ == '__main__':
    process_file('data/test.xml')
