import pandas as pd
import json
import fm


def generate_NRC_VAD_dict(file_path, dict_path):
    df = pd.read_csv(file_path, sep='\t')
    NRC = {}
    for i, row in df.iterrows():
        if row[1] != 'NO TRANSLATION':
            NRC[row[1]] = tuple(row[2:])
        else:
            NRC[row[0]] = tuple(row[2:])
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(NRC, ensure_ascii=False))


def get_emotion_intensity(NRC, word, lam):
    if word not in NRC:
        return -1
    v, a, d = NRC[word]
    w = lam * v + (1 - lam) * a
    return w


if __name__ == "__main__":
# generate_NRC_VAD_dict('data/NRC_VAD/Chinese (Simplified)-zh-CN-NRC-VAD-Lexicon.txt','data/NRC_VAD/NRC.json')
