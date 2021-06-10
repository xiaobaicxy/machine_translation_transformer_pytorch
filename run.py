# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import copy

from data_processor import DataProcessor
from transformer import Transformer
from loss import MaskCriterion

def test(mode, data, seq_len, device):
    model.eval()
    total_words = 0
    total_loss = 0.
    with torch.no_grad():
        for i, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            batch_x = torch.from_numpy(batch_x).to(device).long() 
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            
            batch_predicts = model(batch_x, batch_y_decoder_input, batch_x_len, batch_y_len)
            
            # 生成masks：
            max_seq_len = np.arange(seq_len-1)
            max_seq_len = torch.from_numpy(max_seq_len).to(device).long()
            batch_target_masks = max_seq_len[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks, device)
            
            num_words = torch.sum(batch_y_len).item()
            total_loss += loss.item() * num_words
            total_words += num_words
        test_loss = total_loss/total_words
        print("Test Loss:", test_loss)
    return test_loss

def train(model, data, loss_func, optimizer, epoches, seq_len, device):
    test_datasets = []
    min_test_loss = 1
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epoches):
        model.train()
        total_words = 0
        total_loss = 0.
        for it, (batch_x, batch_x_len, batch_y, batch_y_len) in enumerate(data):
            # 创建验证数据集
            if(epoch == 0 and it % 5 == 0):
                test_datasets.append((batch_x, batch_x_len, batch_y, batch_y_len))
                continue
            batch_x = torch.from_numpy(batch_x).to(device).long()
            batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
            
            # 因为训练时，decoder根据上一步的输出（预测词）和encoder的输出，以及上一步输出对应的实际词预测下一个词
            # 所以输入到decoder中的目标语句为[BOS, word_1, word_2, ..., word_n]
            # 预测词对应的实际标签为[word_1, word_2, ..., word_n, EOS]
            batch_y_decoder_input = torch.from_numpy(batch_y[:, :-1]).to(device).long()
            batch_targets = torch.from_numpy(batch_y[:, 1:]).to(device).long()
            batch_y_len = torch.from_numpy(batch_y_len-1).to(device).long()
            batch_y_len[batch_y_len<=0] = 1
            batch_predicts = model(batch_x, batch_y_decoder_input, batch_x_len, batch_y_len)

            # 生成masks：
            max_seq_len = np.arange(seq_len-1) # 因为shift，所以减1 (直接用torch.arange报错了)
            max_seq_len = torch.from_numpy(max_seq_len).to(device).long()
            batch_target_masks = max_seq_len[None, :] < batch_y_len[:, None]
            batch_target_masks = batch_target_masks.float()
            loss = loss_func(batch_predicts, batch_targets, batch_target_masks, device)
            
            num_words = torch.sum(batch_y_len).item() # 每个batch总的词量
            total_loss += loss.item() * num_words
            total_words += num_words

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.) # 防止梯度爆炸
            optimizer.step()
            
        if( (epoch + 1) % 1000 == 0):
            print("Epoch {} / {}, Train Loss: {}".format(epoch+1, epoches, total_loss/total_words))
            test_loss = test(model, test_datasets, seq_len, device)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

def en2cn_translate(model, 
                    sentence_id, 
                    en_datas, 
                    cn_datas, 
                    cn_dict, 
                    inv_en_dict, 
                    inv_cn_dict, 
                    max_x_len, 
                    max_y_len, 
                    processor, 
                    device):
    # 英文翻译成中文
    en_sentence = " ".join([inv_en_dict[idx] for idx in en_datas[sentence_id]]) # 英文句子
    cn_sentence = " ".join([inv_cn_dict[idx] for idx in cn_datas[sentence_id]]) # 对应实际的中文句子

    batch_x, batch_x_len = processor.add_padding([en_datas[sentence_id]], max_x_len)

    batch_x = torch.from_numpy(batch_x).to(device).long()
    batch_x_len = torch.from_numpy(batch_x_len).to(device).long()
    
    # 第一个时间步的前项输出
    bos = torch.Tensor([[cn_dict["BOS"]]]).to(device).long()
    
    translation = model.translate(batch_x, bos, batch_x_len, max_x_len, max_y_len)
    translation = [inv_cn_dict[i] for i in translation.data.cpu().numpy().reshape(-1)] #index2word
    trans = []
    for word in translation:
        if(word != "EOS"):
            trans.append(word)
        else:
            trans.append(word)
            break
    print(en_sentence)
    print(cn_sentence)
    print(" ".join(trans))

class Config:
    def __init__(self):
        # model 参数
        self.src_vocab_size = 50000
        self.tar_vocab_size = 50000
        self.d_model = 64
        self.num_head = 8
        self.seq_len = 32
        self.hidden_size = 128
        self.dropout = 0.2
        self.num_classes = 2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_encoder_layer = 2
        self.num_decoder_layer = 2
        
        # 训练参数
        self.batch_size = 64
        self.lr = 0.0002
        self.epochs = 20000
        
        # 数据路径
        self.src_language_path = "./dataset/translation/news-commentary-v12.zh-en.en"
        self.tar_language_path = "./dataset/translation/news-commentary-v12.zh-en.zh"

if __name__ == "__main__":
    config = Config()
    processor = DataProcessor()

    en_datas = processor.load_data(config.src_language_path)
    cn_datas = processor.load_data(config.tar_language_path, is_en=False)

    en_word2index, total_en_words = processor.word2index(en_datas, config.src_vocab_size)
    config.src_vocab_size = total_en_words
    cn_word2index, total_cn_words = processor.word2index(cn_datas, config.tar_vocab_size)
    config.tar_vocab_size = total_cn_words

    en_sentence_encoding, cn_sentence_encoding = processor.encode(en_datas, cn_datas, en_word2index, cn_word2index)
    datasets = processor.generate_dataset(en_sentence_encoding, cn_sentence_encoding, config.batch_size, config.seq_len)

    model = Transformer(config).to(config.device)
    loss_func = MaskCriterion().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model = train(model, datasets, loss_func, optimizer, config.epochs, config.seq_len, config.device)
    
    en_index2word = processor.index2word(en_word2index)
    cn_index2word = processor.index2word(cn_word2index)

    sentence_id = 0
    en2cn_translate(model, 
                    sentence_id, 
                    en_sentence_encoding, 
                    cn_sentence_encoding, 
                    cn_word2index, 
                    en_index2word, 
                    cn_index2word, 
                    config.seq_len, 
                    config.seq_len-1, 
                    processor, 
                    config.device)
