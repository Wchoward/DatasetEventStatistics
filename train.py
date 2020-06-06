import tensorflow as tf
from tensorflow.keras import layers
from model import MainModel
import numpy as np
import os
import time
import fm
# import word2vector
from sklearn.metrics import precision_recall_fscore_support

# import tensorflow.contrib.eager as tfe

os.environ['CUDA_VISIBLE_DEVICES'] = "7"

EPOCH = 8
BATCH_SIZE = 16


# tfe.enable_eager_execution()


def train_per_emotion(path, module1, module2):
    original_text = np.load(os.path.join(path, 'origin_text.npy'), allow_pickle=True)
    # SAMPLE_NUM = len(original_text)
    # original_text = tf.convert_to_tensor(original_text, dtype=tf.float32)
    original_text = original_text.astype(np.float32)
    cause_events = np.load(os.path.join(path, 'cause_event.npy'), allow_pickle=True)
    # cause_events = tf.convert_to_tensor(cause_events, dtype=tf.float32)
    cause_events = cause_events.astype(np.float32)
    labels = np.load(os.path.join(path, 'if_cause.npy'), allow_pickle=True)
    # labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # labels = np.one_hot(labels, depth=2)
    labels = labels.astype(np.int32)
    labels = np.eye(2)[labels]

    with tf.name_scope("input_module"):
        original_text_input = layers.Input(batch_shape=(None, 169, 300))
        event_input = layers.Input(batch_shape=(None, 44, 300))
        # predictions = layers.Input(batch_shape=(None, 1))

    net = MainModel()
    output = net.model_build(module1, module2, original_text_input, event_input)

    model = tf.keras.Model(inputs=[original_text_input, event_input], outputs=output)
    # model = tf.keras.Model()

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    '''
    train_dataset1 = tf.data.Dataset.from_tensor_slices((original_text[400:], cause_events[400:]))
    train_dataset2 = tf.data.Dataset.from_tensor_slices(labels[400:])
    train_dataset = tf.data.Dataset.zip((train_dataset1, train_dataset2)).repeat(10).shuffle(100).batch(16)
    
    test_dataset1 = tf.data.Dataset.from_tensor_slices((original_text[:400], cause_events[:400]))
    test_dataset2 = tf.data.Dataset.from_tensor_slices(labels[:400])
    test_dataset = tf.data.Dataset.zip((test_dataset1, test_dataset2)).repeat(10).shuffle(100)
    '''

    train_original_text = original_text[400:]
    train_cause_events = cause_events[400:]
    train_labels = labels[400:]
    test_original_text = original_text[:400]
    test_cause_events = cause_events[:400]
    test_labels = labels[:400]

    model.fit(x=[train_original_text, train_cause_events], y=train_labels, epochs=EPOCH, shuffle=True,
              validation_data=([test_original_text, test_cause_events], test_labels), batch_size=16)

    cur_time = time.localtime(time.time())
    # model.save(os.path.join('model/my_model',
    #                         'my_model_{}_{}_{}_{}_{}.h5'.format(cur_time.tm_mon, cur_time.tm_mday, cur_time.tm_hour,
    #                                                             cur_time.tm_min, cur_time.tm_sec)))

    prediction = model.predict([test_original_text, test_cause_events])
    true_labels = np.argmax(test_labels, axis=1)
    pred_labels = np.argmax(prediction, axis=1)

    p_r_f = precision_recall_fscore_support(y_true=true_labels, y_pred=pred_labels, average='binary')
    p_r_f = [round(x, 3) for x in p_r_f[:3]]
    print(p_r_f)
    return p_r_f


def get_dir_list(dir_path):
    lst = []
    for s in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, s)):
            lst.append(s)
    return lst


def train_lst(root_path, category_lst, module):
    res = []
    for module1 in module:
        for module2 in module:
            res.append(module1 + '+' + module2 + ',,,')
            res.append('Emotion,P,R,F')
            for category in category_lst:
                path = os.path.join(root_path, category)
                category_p_r_f = [category] + train_per_emotion(path, module1, module2)
                s_category_p_r_f = ','.join(str(i) for i in category_p_r_f)
                res.append(s_category_p_r_f)
            res.append('')
    return res


if __name__ == '__main__':
    root_path = 'data/origin_w2v_emotion_category'
    module = ['cnn', 'lstm', 'attention']
    category_lst = get_dir_list(root_path)
    # train_per_emotion('data/merge_expand_emotion_embedding_1', 'attention', 'attention')
    # origin w2v
    # result = train_lst(root_path, category_lst, module)
    # fm.save_file('result/origin_result.txt', result)
    # lambda=0
    root_path = 'data/expand_w2v_emotion_category_lambda_0'
    result = train_lst(root_path, category_lst, module)
    fm.save_file('result/expand_lambda_result_0.txt', result)
    # lambda=0.25
    root_path = 'data/expand_w2v_emotion_category_lambda_0.25'
    result = train_lst(root_path, category_lst, module)
    fm.save_file('result/expand_lambda_result_0.25.txt', result)
    # lambda=0.5
    root_path = 'data/expand_w2v_emotion_category_lambda_0.5'
    result = train_lst(root_path, category_lst, module)
    fm.save_file('result/expand_lambda_result_0.5.txt', result)
    # lambda=0.75
    root_path = 'data/expand_w2v_emotion_category_lambda_0.75'
    result = train_lst(root_path, category_lst, module)
    fm.save_file('result/expand_lambda_result_0.75.txt', result)
    # lambda=1
    root_path = 'data/expand_w2v_emotion_category_lambda_1'
    result = train_lst(root_path, category_lst, module)
    fm.save_file('result/expand_lambda_result_1.txt', result)
