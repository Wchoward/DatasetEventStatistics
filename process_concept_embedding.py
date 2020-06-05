import fm
import process_NRC_VAD

from word2vector import load_word2vec_model_text_to_dict, save_dict_to_word2vec_model_text


def get_concept_words(concept_dict_path):
    """
    从conceptDict中抽取出 含有 concept的所有相关concept单词list以及简化版的dict
    :param concept_dict_path:
    :return:
    """
    concept_dict = fm.load_dict_json(concept_dict_path)
    lst = []
    simplified_concept_dict = {}
    for key, vlst in concept_dict.items():
        concept_lst = []
        lst.append(key)
        concept_lst.append({'Weight': 1.0, 'Concept': key})
        for i in vlst:
            dic = {'Weight': i['Weight']}
            if i['Concept1'] != key:
                dic['Concept'] = i['Concept1']
                lst.append(i['Concept1'])
                concept_lst.append(dic)
            if i['Concept2'] != key:
                dic['Concept'] = i['Concept2']
                lst.append(i['Concept2'])
                concept_lst.append(dic)
        simplified_concept_dict[key] = concept_lst
    lst = list(set(lst))
    return lst, simplified_concept_dict


def get_expand_concept_words(concept_dict_path):
    """
    从conceptDict中抽取出 含有 concept的所有相关concept单词list以及简化版的dict
    :param concept_dict_path:
    :return:
    """
    concept_dict = fm.load_dict_json(concept_dict_path)
    lst = []
    simplified_concept_dict = {}
    for key, vlst in concept_dict.items():
        concept_lst = []
        lst.append(key)
        concept_lst.append({'Weight': 1.0, 'Concept': key})
        tmp = {}
        word = ''
        for l in vlst[:4]:
            if l['Concept1'] in tmp.keys():
                tmp[l['Concept1']] += 1
            else:
                tmp[l['Concept1']] = 1
            if l['Concept2'] in tmp.keys():
                tmp[l['Concept2']] += 1
            else:
                tmp[l['Concept2']] = 1
        for k, v in tmp.items():
            if v == len(tmp):
                word = k
                break
        if word != '':
            concept_lst.append({'Weight': 1.0, 'Concept': word})
        for i in vlst:
            dic = {'Weight': i['Weight']}
            if i['Concept1'] != word:
                dic['Concept'] = i['Concept1']
                lst.append(i['Concept1'])
                concept_lst.append(dic)
            if i['Concept2'] != word:
                dic['Concept'] = i['Concept2']
                lst.append(i['Concept2'])
                concept_lst.append(dic)
        simplified_concept_dict[key] = concept_lst
    lst = list(set(lst))
    return lst, simplified_concept_dict


def filter_conceptnet_embedding(dict_path, file_path_in, file_path_out):
    """
    缩减conceptNet embedding规模到与我们语料抽取出的concept相同
    :param dict_path:
    :param file_path_in:
    :param file_path_out:
    :return:
    """
    lst, dic = get_concept_words(dict_path)
    full_dict = {}
    filter_dict = {}
    with open(file_path_in, encoding='utf-8') as f:
        next(f)
        for line in f:
            tmp = line.strip(' \n').split(' ', 1)
            k = tmp[0]
            v = tmp[1]
            full_dict[k] = v
    for word in lst:
        if word in full_dict:
            filter_dict[word] = full_dict[word]
    print('filter conceptNet length:', len(filter_dict))
    words = list(filter_dict.keys())
    vectors = list(filter_dict.values())
    embedding_lst = [str(len(filter_dict)) + ' 300']
    for i in range(len(filter_dict)):
        embedding_lst.append(words[i] + ' ' + vectors[i])
    fm.save_file(file_path_out, embedding_lst)


# def weighted_conceptnet_embedding(concept_dict, embedding_dict, lam):
#     weighted_model_dict = {}
#     concept_words = list(concept_dict.keys())
#     for word in concept_words:


if __name__ == "__main__":
    lst, concept_dict = get_expand_concept_words('data/conceptNet/expand_conceptDict.json')
    # embedding_dict = load_word2vec_model_text_to_dict('model/conceptnet_embedding/filtered_conceptnet_embedding.txt')
    # fm.save_file('data/conceptNet/expand_conceptWords.txt', lst)
    fm.save_dict_json('data/conceptNet/simplified_expand_concept_dict.json', concept_dict)
    # filter_conceptnet_embedding('data/conceptNet/expand_conceptDict.json',
    #                             'model/conceptnet_embedding/zhs_conceptnet_embedding.txt',
    #                             'model/conceptnet_embedding/filtered_expand_conceptnet_embedding.txt')
