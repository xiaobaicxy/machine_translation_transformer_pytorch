# -*- coding: utf-8 -*-
import os
import numpy as np
import jieba
import nltk
from collections import Counter

np.random.seed(123) # 保证每次运行生成的随机数相同
class DataProcessor:
    def load_data(self, file_name, is_en=True):
        # 逐句读取文本，并将句子进行分词，且在句子前面加上'BOS'表示句子开始，在句子末尾加上'EOS'表示句子结束
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if(i>10): # for debug
                   break
                line = line.strip()
                if(is_en):
                    datas.append(["BOS"] + nltk.word_tokenize(line.lower()) + ["EOS"])
                else:
                    datas.append(["BOS"] + list(jieba.cut(line, cut_all=False)) + ["EOS"])
        return datas

    def word2index(self, sentences, max_words):
        # 统计文本中每个词出现的频数，并用出现次数最多的max_words个词创建词典，
        # 且在词典中加入'UNK'表示词典中未出现的词，'PAD'表示后续句子中添加的padding（保证每个batch中的句子等长）
        word_count = Counter()
        for sentence in sentences:
            for word in sentence:
                word_count[word] += 1
        
        most_common_words = word_count.most_common(max_words)  # 最常见的max_words个词
        word_dict = {w[0]: index+2 for index, w in enumerate(most_common_words)}  # word2index
        word_dict["PAD"] = 0
        word_dict["UNK"] = 1
        total_words = len(most_common_words) + 2  # 总词量（+2：词典中添加了“UNK”和“PAD”）
        return word_dict, total_words

    def index2word(self, word_dict):
        inv_dict = {v: k for k, v in word_dict.items()}
        return inv_dict

    def encode(self, en_sentences, cn_sentences, en_dict, cn_dict):
        # 句子编码：将句子中的词转换为词表中的index
        
        # 不在词典中的词用”UNK“表示
        out_en_sentences = [[en_dict.get(w, en_dict['UNK']) for w in sentence] for sentence in en_sentences]
        out_cn_sentences = [[cn_dict.get(w, cn_dict['UNK']) for w in sentence] for sentence in cn_sentences]
        
        return out_en_sentences, out_cn_sentences

    def get_batches(self, num_sentences, batch_size, shuffle=True):
        # 用每个句子在原始文本中的（位置）行号创建每个batch的数据索引
        sentence_idx = np.arange(start=0, stop=num_sentences, step=1)
        batch_first_idx = np.arange(start=0, stop=num_sentences, step=batch_size) # 每个batch中第一个句子在文本中的位置（行号）
        if(shuffle):
            np.random.shuffle(sentence_idx)
        
        batches = []
        for first_idx in batch_first_idx:
            batch = np.array([sentence_idx[i] for i in range(first_idx, min(first_idx+batch_size, num_sentences))]).astype('int32')
            batches.append(batch)
        return batches

    def add_padding(self, batch_sentences, max_len):
        # 为每个batch的数据添加padding，并记录下句子原本的长度
        lengths = [len(sentence)  if len(sentence) < max_len else max_len for sentence in batch_sentences] # 每个句子的实际长度，过长句子截断
        data = []
        for sentence in batch_sentences:
            sen_len = len(sentence)
            if sen_len > max_len:
                sentence = sentence[:max_len]
            # 将每个句子末尾添0，使得每个batch中的句子等长（后续将每个batch数据转换成tensor时，每个batch中的数据维度必须一致）
            sentence = sentence + [0]*(max_len - sen_len) 
            data.append(sentence)
        data = np.array(data).astype('int32')
        data_lengths = np.array(lengths).astype('int32')
        return data, data_lengths
    
    def generate_dataset(self, en, cn, batch_size, max_len):
        #生成数据集
        batches = self.get_batches(len(en), batch_size)
        datasets = []
        for batch in batches:
            batch_en = [en[idx] for idx in batch]
            batch_cn = [cn[idx] for idx in batch]
            batch_x, batch_x_len = self.add_padding(batch_en, max_len)
            batch_y, batch_y_len = self.add_padding(batch_cn, max_len)
            datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
        return datasets
