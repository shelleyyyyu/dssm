#!/usr/bin/env python
# encoding=utf-8
import sys
from inspect import getblock
import json
import os
from os import read
from numpy.core.fromnumeric import mean
import numpy as np
import paddlehub as hub
import six
import math
import random
import sys
from util import read_file
from config import Config
# 配置文件
conf = Config()


class Vocabulary(object):
    def __init__(self, meta_file, max_len, allow_unk=0, unk="$UNK$", pad="$PAD$",):
        self.voc2id = {}
        self.id2voc = {}
        self.unk = unk
        self.pad = pad
        self.max_len = max_len
        self.allow_unk = allow_unk
        with open(meta_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = convert_to_unicode(line.strip("\n"))
                self.voc2id[line] = i
                self.id2voc[i] = line
        self.size = len(self.voc2id)
        self.oov_num = self.size + 1

    def fit(self, words_list):
        """
        :param words_list: [[w11, w12, ...], [w21, w22, ...], ...]
        :return:
        """
        word_lst = []
        word_lst_append = word_lst.append
        for words in words_list:
            if not isinstance(words, list):
                print(words)
                continue
            for word in words:
                word = convert_to_unicode(word)
                word_lst_append(word)
        word_counts = Counter(word_lst)
        if self.max_num_word < 0:
            self.max_num_word = len(word_counts)
        sorted_voc = [w for w, c in word_counts.most_common(self.max_num_word)]
        self.max_num_word = len(sorted_voc)
        self.oov_index = self.max_num_word + 1
        self.voc2id = dict(zip(sorted_voc, range(1, self.max_num_word + 1)))
        return self

    def _transform2id(self, word):
        word = convert_to_unicode(word)
        if word in self.voc2id:
            return self.voc2id[word]
        elif self.allow_unk:
            return self.voc2id[self.unk]
        else:
            print(word)
            raise ValueError("word:{} Not in voc2id, please check".format(word))

    def _transform_seq2id(self, words, padding=0):
        out_ids = []
        words = convert_to_unicode(words)
        if self.max_len:
            words = words[:self.max_len]
        for w in words:
            out_ids.append(self._transform2id(w))
        if padding and self.max_len:
            while len(out_ids) < self.max_len:
                out_ids.append(0)
        return out_ids
    
    def _transform_intent2ont_hot(self, words, padding=0):
        # 将多标签意图转为 one_hot
        out_ids = np.zeros(self.size, dtype=np.float32)
        words = convert_to_unicode(words)
        for w in words:
            out_ids[self._transform2id(w)] = 1.0
        return out_ids

    def _transform_seq2bert_id(self, words, padding=0):
        out_ids, seq_len = [], 0
        words = convert_to_unicode(words)
        if self.max_len:
            words = words[:self.max_len]
        seq_len = len(words)
        # 插入 [CLS], [SEP]
        out_ids.append(self._transform2id("[CLS]"))
        for w in words:
            out_ids.append(self._transform2id(w))
        mask_ids = [1 for _ in out_ids]
        if padding and self.max_len:
            while len(out_ids) < self.max_len + 1:
                out_ids.append(0)
                mask_ids.append(0)
        seg_ids = [0 for _ in out_ids]
        return out_ids, mask_ids, seg_ids, seq_len

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _transform_2seq2bert_id(self, seq1, seq2, padding=0):
        out_ids, seg_ids, seq_len = [], [1], 0
        seq1 = [x for x in convert_to_unicode(seq1)]
        seq2 = [x for x in convert_to_unicode(seq2)]
        # 截断
        self._truncate_seq_pair(seq1, seq2, self.max_len - 2)
        # 插入 [CLS], [SEP]
        out_ids.append(self._transform2id("[CLS]"))
        for w in seq1:
            out_ids.append(self._transform2id(w))
            seg_ids.append(0)
        for w in seq2:
            out_ids.append(self._transform2id(w))
            seg_ids.append(1)
        mask_ids = [1 for _ in out_ids]
        if padding and self.max_len:
            while len(out_ids) < self.max_len + 1:
                out_ids.append(0)
                mask_ids.append(0)
                seg_ids.append(0)
        return out_ids, mask_ids, seg_ids, seq_len

    def transform(self, seq_list, is_bert=0):
        if is_bert:
            return [self._transform_seq2bert_id(seq) for seq in seq_list]
        else:
            return [self._transform_seq2id(seq) for seq in seq_list]

    def __len__(self):
        return len(self.voc2id)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def gen_word_set(file_path, out_path='./data/words.txt'):
    word_set = set()
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = [prefix, title]
            query_pred = json.loads(query_pred)
            for w in prefix:
                word_set.add(w)
            for each in query_pred:
                for w in each:
                    word_set.add(w)
    with open(word_set, 'w', encoding='utf-8') as o:
        for w in word_set:
            o.write(w + '\n')
    pass

def convert_word2id(max_seq_len, query, vocab_map, unk='[UNK]', pad='[PAD]'):
    ids = []
    for w in query:
        if w in vocab_map:
            ids.append(vocab_map[w])
        else:
            ids.append(vocab_map[conf.unk])
    while len(ids) < max_seq_len:
        ids.append(vocab_map[conf.pad])
    return ids[:max_seq_len]

def convert_seq2bow(query, vocab_map):
    bow_ids = np.zeros(conf.nwords)
    for w in query:
        if w in vocab_map:
            bow_ids[vocab_map[w]] += 1
        else:
            bow_ids[vocab_map[conf.unk]] += 1
    return bow_ids

def get_data(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    #蒲公英	{"蒲公英根": "0.013", "蒲公英茶可以天天喝吗": "0.013", "蒲公英茶的功效与作用": "0.130", "蒲公英根泡水喝的功效": "0.025", "蒲公英之恋": "0.010", "蒲公英图片": "0.059", "蒲公英的功效": "0.018", "蒲公英泡水喝的功效": "0.018", "蒲公英茶": "0.057", "蒲公英的功效与作用": "0.113"}	蒲公英	应用
    data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr, cur_len = [], []
            query_pred = json.loads(query_pred)
            # only 4 negative sample
            for each in query_pred:
                if each == title:
                    continue
                cur_arr.append(convert_word2id(each, conf.vocab_map))
                each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                cur_len.append(each_len)
            if len(cur_arr) >= 4:
                data_map['query'].append(convert_word2id(prefix, conf.vocab_map))
                data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map))
                data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                data_map['doc_neg'].extend(cur_arr[:4])
                data_map['doc_neg_len'].extend(cur_len[:4])
            pass
    return data_map

def get_article_data(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    #蒲公英	{"蒲公英根": "0.013", "蒲公英茶可以天天喝吗": "0.013", "蒲公英茶的功效与作用": "0.130", "蒲公英根泡水喝的功效": "0.025", "蒲公英之恋": "0.010", "蒲公英图片": "0.059", "蒲公英的功效": "0.018", "蒲公英泡水喝的功效": "0.018", "蒲公英茶": "0.057", "蒲公英的功效与作用": "0.113"}	蒲公英	应用
    data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
    with open(file_path, encoding='utf8') as f:
        raw_data = f.readlines()
        pos_neg_group = int(len(raw_data)/5)
        for count in range(pos_neg_group):
            if len(raw_data[count*5].strip().split('\t')) != 3 \
                and raw_data[(count*5)+1].strip().split('\t') != 3 \
                and raw_data[(count*5)+2].strip().split('\t') != 3 \
                and raw_data[(count*5)+3].strip().split('\t') != 3 \
                and raw_data[(count*5)+4].strip().split('\t') != 3:
                continue
            pos_q, pos_doc, pos_label = raw_data[count*5].strip().split('\t')
            neg_1_q, neg_1_doc, neg_1_label = raw_data[(count*5)+1].strip().split('\t')
            neg_2_q, neg_2_doc, neg_2_label = raw_data[(count*5)+2].strip().split('\t')
            neg_3_q, neg_3_doc, neg_3_label = raw_data[(count*5)+3].strip().split('\t')
            neg_4_q, neg_4_doc, neg_4_label = raw_data[(count*5)+4].strip().split('\t')
            cur_arr, cur_len = [], []
            # only 4 negative sample
            cur_arr.append(convert_word2id(conf.max_doc_seq_len, neg_1_doc, conf.vocab_map))
            cur_len.append(len(neg_1_doc) if len(neg_1_doc) < conf.max_doc_seq_len else conf.max_doc_seq_len)
            cur_arr.append(convert_word2id(conf.max_doc_seq_len, neg_2_doc, conf.vocab_map))
            cur_len.append(len(neg_2_doc) if len(neg_2_doc) < conf.max_doc_seq_len else conf.max_doc_seq_len)
            cur_arr.append(convert_word2id(conf.max_doc_seq_len, neg_3_doc, conf.vocab_map))
            cur_len.append(len(neg_3_doc) if len(neg_3_doc) < conf.max_doc_seq_len else conf.max_doc_seq_len)
            cur_arr.append(convert_word2id(conf.max_doc_seq_len, neg_4_doc, conf.vocab_map))
            cur_len.append(len(neg_4_doc) if len(neg_4_doc) < conf.max_doc_seq_len else conf.max_doc_seq_len)

            if len(cur_arr) >= 4:
                data_map['query'].append(convert_word2id(conf.max_query_seq_len, pos_q, conf.vocab_map))
                data_map['query_len'].append(len(pos_q) if len(pos_q) < conf.max_query_seq_len else conf.max_query_seq_len)
                data_map['doc_pos'].append(convert_word2id(conf.max_doc_seq_len, pos_doc, conf.vocab_map))
                data_map['doc_pos_len'].append(len(pos_doc) if len(pos_doc) < conf.max_doc_seq_len else conf.max_doc_seq_len)
                data_map['doc_neg'].extend(cur_arr[:4])
                data_map['doc_neg_len'].extend(cur_len[:4])
            pass
    return data_map

def get_data_siamese_rnn(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
    """
    data_arr = []
    # 蒲公英	{"蒲公英根": "0.013", "蒲公英茶可以天天喝吗": "0.013", "蒲公英茶的功效与作用": "0.130", "蒲公英根泡水喝的功效": "0.025", "蒲公英之恋": "0.010", "蒲公英图片": "0.059", "蒲公英的功效": "0.018", "蒲公英泡水喝的功效": "0.018", "蒲公英茶": "0.057", "蒲公英的功效与作用": "0.113"}	蒲公英	应用
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, _, title, tag, label = spline
            prefix_seq = convert_word2id(prefix, conf.vocab_map)
            title_seq = convert_word2id(title, conf.vocab_map)
            data_arr.append([prefix_seq, title_seq, int(label)])
    return data_arr

def get_data_bow(file_path):
    """
    gen datasets, convert word into word ids.
    :param file_path:
    :return: [[query, prefix, label]], shape = [n, 3]
    """
    data_arr = []
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            # 蒲公英	{"蒲公英根": "0.013", "蒲公英茶可以天天喝吗": "0.013", "蒲公英茶的功效与作用": "0.130", "蒲公英根泡水喝的功效": "0.025", "蒲公英之恋": "0.010", "蒲公英图片": "0.059", "蒲公英的功效": "0.018", "蒲公英泡水喝的功效": "0.018", "蒲公英茶": "0.057", "蒲公英的功效与作用": "0.113"}	蒲公英	应用
            prefix, _, title, tag, label = spline
            prefix_ids = convert_seq2bow(prefix, conf.vocab_map)
            title_ids = convert_seq2bow(title, conf.vocab_map)
            data_arr.append([prefix_ids, title_ids, int(label)])
    return data_arr

def trans_lcqmc(dataset):
    """
    最大长度
    """
    out_arr, text_len =  [], []
    for each in dataset:
        t1, t2, label = each.text_a, each.text_b, int(each.label)
        t1_ids = convert_word2id(t1, conf.vocab_map)
        t1_len = conf.max_seq_len if len(t1) > conf.max_seq_len else len(t1)
        t2_ids = convert_word2id(t2, conf.vocab_map)
        t2_len = conf.max_seq_len if len(t2) > conf.max_seq_len else len(t2)
        # t2_len = len(t2)
        out_arr.append([t1_ids, t1_len, t2_ids, t2_len, label])
        # out_arr.append([t1_ids, t1_len, t2_ids, t2_len, label, t1, t2])
        text_len.extend([len(t1), len(t2)])
        pass
    print("max len", max(text_len), "; avg len", '%.4f'%mean(text_len), "; cover rate:", np.mean([x <= conf.max_seq_len for x in text_len]))
    return out_arr

def trans_article(config, fname):
    """
    最大长度
    """
    out_arr, text_q_len, text_d_len = [], [], []
    with open(fname, 'r', encoding='utf-8') as file:
        data = file.readlines()
        for each in data:
            tmp_arr = each.strip().split('\t')
            if len(tmp_arr) != 3:
                continue
            t1, t2, label = tmp_arr[0], tmp_arr[1], tmp_arr[2]
            t1_ids = convert_word2id(config['max_query_seq_len'], t1, conf.vocab_map)
            t1_len = config['max_query_seq_len'] if len(t1) > config['max_query_seq_len'] else len(t1)
            t2_ids = convert_word2id(config['max_doc_seq_len'], t2, conf.vocab_map)
            t2_len = config['max_doc_seq_len'] if len(t2) > config['max_doc_seq_len'] else len(t2)
            # t2_len = len(t2)
            out_arr.append([t1_ids, t1_len, t2_ids, t2_len, label])
            # out_arr.append([t1_ids, t1_len, t2_ids, t2_len, label, t1, t2])
            text_d_len.extend([len(t2)])
            text_q_len.extend([len(t1)])
            pass
        print("Q: max len", max(text_q_len), "; avg len", '%.4f'%mean(text_q_len), "; cover rate:", np.mean([x <= config['max_query_seq_len'] for x in text_q_len]))
        print("D: max len", max(text_d_len), "; avg len", '%.4f'%mean(text_d_len), "; cover rate:", np.mean([x <= config['max_doc_seq_len'] for x in text_d_len]))
    return out_arr

def get_article(config):
    train_fname = os.path.join(config['dataset_dir'], config['dataset_prefix']+'train.txt')
    valid_fname = os.path.join(config['dataset_dir'], config['dataset_prefix']+'valid.txt')
    test_fname = os.path.join(config['dataset_dir'], config['dataset_prefix']+'test.txt')
    train_set = trans_article(config, train_fname)
    dev_set = trans_article(config, valid_fname)
    test_set = trans_article(config, test_fname)
    return train_set, dev_set, test_set
    # return test_set, test_set, test_set

def get_lcqmc():
    """
    使用LCQMC数据集，并将其转为word_id
    """
    #['喜欢打篮球的男生喜欢什么样的女生', 16, '爱打篮球的男生喜欢什么样的女生', 15, 1]
    dataset = hub.dataset.LCQMC()
    train_set = trans_lcqmc(dataset.train_examples)
    dev_set = trans_lcqmc(dataset.dev_examples)
    test_set = trans_lcqmc(dataset.test_examples)
    return train_set, dev_set, test_set
    # return test_set, test_set, test_set

def trans_lcqmc_bert(dataset:list, vocab:Vocabulary, is_merge=0):
    """
    最大长度
    """
    out_arr, text_len =  [], []
    for each in dataset:
        t1, t2, label = each.text_a, each.text_b, int(each.label)
        if is_merge:
            out_ids1, mask_ids1, seg_ids1, seq_len1 = vocab._transform_2seq2bert_id(t1, t2, padding=1)
            out_arr.append([out_ids1, mask_ids1, seg_ids1, seq_len1, label])
            text_len.extend([len(t1) + len(t2)])
        else:
            out_ids1, mask_ids1, seg_ids1, seq_len1 = vocab._transform_seq2bert_id(t1, padding=1)
            out_ids2, mask_ids2, seg_ids2, seq_len2 = vocab._transform_seq2bert_id(t2, padding=1)
            out_arr.append([out_ids1, mask_ids1, seg_ids1, seq_len1, out_ids2, mask_ids2, seg_ids2, seq_len2, label])
            text_len.extend([len(t1), len(t2)])
        pass
    print("max len", max(text_len), "avg len", mean(text_len), "cover rate:", np.mean([x <= conf.max_seq_len for x in text_len]))
    return out_arr

def get_lcqmc_bert(vocab:Vocabulary, is_merge=0):
    """
    使用LCQMC数据集，并将每个query其转为word_id，
    """
    dataset = hub.dataset.LCQMC()
    train_set = trans_lcqmc_bert(dataset.train_examples, vocab, is_merge)
    dev_set = trans_lcqmc_bert(dataset.dev_examples, vocab, is_merge)
    test_set = trans_lcqmc_bert(dataset.test_examples, vocab, is_merge)
    return train_set, dev_set, test_set
    # test_set = test_set[:100]
    # return test_set, test_set, test_set

def get_test(file_:str, vocab:Vocabulary):
    test_arr = read_file(file_, '\t') # [[q1, q2],...]
    out_arr = []
    for line in test_arr:
        if len(line) != 3:
            print('wrong line size=', len(line))
        t1, t2, label = line   # [t1_ids, t1_len, t2_ids, t2_len, label]
        t1_ids = vocab._transform_seq2id(t1, padding=1)
        t1_len = vocab.max_len if len(t1) > vocab.max_len else len(t1)
        t2_ids = vocab._transform_seq2id(t2, padding=1)
        t2_len = vocab.max_len if len(t2) > vocab.max_len else len(t2)
        out_arr.append([t1_ids, t1_len, t2_ids, t2_len])
    return out_arr, test_arr

def get_test_bert(file_:str, vocab:Vocabulary, is_merge=0):
    test_arr = read_file(file_, '\t') # [[q1, q2],...]
    out_arr = get_test_bert_by_arr(test_arr, vocab, is_merge)
    return out_arr, test_arr

def get_test_bert_by_arr(test_arr:list, vocab:Vocabulary, is_merge=0):
    # test_arr # [[q1, q2],...]
    out_arr = []
    for line in test_arr:
        if len(line) != 2:
            print('wrong line size=', len(line))
        t1, t2 = line   # [t1_ids, t1_len, t2_ids, t2_len, label]
        if is_merge:
            out_ids1, mask_ids1, seg_ids1, seq_len1 = vocab._transform_2seq2bert_id(t1, t2, padding=1)
            out_arr.append([out_ids1, mask_ids1, seg_ids1, seq_len1])
        else:
            out_ids1, mask_ids1, seg_ids1, seq_len1 = vocab._transform_seq2bert_id(t1, padding=1)
            out_ids2, mask_ids2, seg_ids2, seq_len2 = vocab._transform_seq2bert_id(t2, padding=1)
            out_arr.append([out_ids1, mask_ids1, seg_ids1, seq_len1, out_ids2, mask_ids2, seg_ids2, seq_len2])
    return out_arr, test_arr

def get_batch(dataset, batch_size=None, is_test=0):
    # tf Dataset太难用，不如自己实现
    # https://stackoverflow.com/questions/50539342/getting-batches-in-tensorflow
    # dataset：每个元素是一个特征，[[x1, x2, x3,...], ...], 如果是测试集，可能就没有标签
    if not batch_size:
        batch_size = 32
    if not is_test:
        random.shuffle(dataset)
    steps = int(math.ceil(float(len(dataset)) / batch_size))
    for i in range(steps):
        idx = i * batch_size
        cur_set = dataset[idx: idx + batch_size]
        cur_set = zip(*cur_set)
        yield cur_set


if __name__ == '__main__':
    # prefix, query_prediction, title, tag, label
    # query_prediction 为json格式。
    file_train = './data/oppo_round1_train_20180929.txt'
    file_vali = './data/oppo_round1_vali_20180929.txt'
    # data_train = get_data(file_train)
    # data_train = get_data(file_vali)
    # print(len(data_train['query']), len(data_train['doc_pos']), len(data_train['doc_neg']))
    dataset = get_lcqmc()
    print(dataset[1][:3])
    for each in get_batch(dataset[1][:3], batch_size=2):
        t1_ids, t1_len, t2_ids, t2_len, label = each
        print(each)
    pass
