#!/usr/bin/env python
# encoding=utf-8
'''
Author: 	zhiyang.zzy 
Date: 		2019-09-25 21:59:54
Contact: 	zhiyangchou@gmail.com
FilePath: /dssm/config.py
Desc: 		
'''


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict


class Config(object):
    def __init__(self):
        self.vocab_map = load_vocab(self.vocab_path)
        self.nwords = len(self.vocab_map)

    unk = '[UNK]'
    pad = '[PAD]'
    vocab_path = './data/vocab.txt'
    file_train = './data/article_pos_4neg/article_train.txt' #change
    file_valid = './data/article_pos_4neg/article_valid.txt' #change
    file_test = './data/article_pos_4neg/article_test.txt' #change
    max_query_seq_len = 10
    max_doc_seq_len = 1000
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.001
    decay_step = 2000
    lr_decay = 0.95
    num_epoch = 300
    epoch_no_imprv = 5
    optimizer = "lazyadam"
    summaries_dir = './results/Summaries/'
    gpu = 0
    word_dim = 100
    batch_size = 64
    keep_prob = 0.5
    dropout = 1- keep_prob
    query_BS = 64 #change

    # checkpoint_dir
    checkpoint_dir='./results/checkpoint'


if __name__ == '__main__':
    conf = Config()
    print(len(conf.vocab_map))
    pass
