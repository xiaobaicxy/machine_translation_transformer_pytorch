# -*- coding: utf-8 -*-
"""
机器翻译 Transformer 算法
@author:
"""
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from data_processor import DataProcessor

# 保证每次运行生成的随机数相同
torch.manual_seed(123)
torch.cuda.manual_seed(123)

def create_atten_masks(x_len, y_len, max_x_len, max_y_len, device):
    # x_len, y_len: [batch_size]
    # max_x_len, max_y_len: int

    max_x_len = torch.arange(max_x_len, device=device)
    x_masks = max_x_len[None, :] < x_len[:, None]

    max_y_len = torch.arange(max_y_len, device=device)
    y_masks = max_y_len[None, :] < y_len[:, None] # A[None,:] op B[:,None]表示给A增加一个dim0,其维度与B的dim0相同
    
    # # masked_fill_填充的是True所在的维度，所以取反(~)
    masks = (~(y_masks.unsqueeze(2) * x_masks.unsqueeze(1))).byte()
    return masks   # [batch_size, max_y_len, max_x_len]


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, pretrained_embedding=None):
        super(InputEmbedding, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        out = self.embedding(x) # [batch_size, seq_len, d_model]
        return out


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.tensor(
            [
                [
                    pos / pow(10000.0, (i//2 * 2.0) / d_model) 
                    for i in range(d_model)
                ] 
            for pos in range(seq_len)]
            ) # 注意 i//2 * 2
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2]) 
        
    def forward(self):
        pe = nn.Parameter(self.pe, requires_grad=False) # [seq_len, d_model]
        return pe


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, device, dropout):
        super(Embedding, self).__init__()
        self.seq_len = seq_len

        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, is_translation=False, cur_step=0):
        # x: [batch_size, seq_len]
        input_embedding = self.input_embedding(x).to(self.device) # [batch_size, seq_len, d_model]
        position_embedding = self.position_embedding().to(self.device) # [seq_len, d_model]
        if is_translation: # 翻译时，只取当前时间步的位置编码
            assert cur_step < self.seq_len
            out = input_embedding + position_embedding[cur_step, :]
        else:
            out = input_embedding + position_embedding # [batch_size, seq_len, d_model]
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k = None):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, masks):
        # Q, K, V:  [batch_size * num_head, seq_len, d_head]
        # masks: [batch_size, q_len, k_len]
        # k_len = v_len
        
        attention_context = torch.matmul(Q, K.permute(0, 2, 1)) # [batch_size * num_head, q_len, k_len]
        if self.d_k is not None:
            attention_context = attention_context / (self.d_k ** 0.5)

        attention_context.data.masked_fill_(masks.bool(), -1e-10)
       
        attention_w = F.softmax(attention_context, dim=2)
        context = torch.matmul(attention_w, V) # [batch_size * num_head, q_len, d_head]
        return context
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, device, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.device = device
        assert d_model % num_head == 0
        self.d_head = d_model // num_head
        self.proj_Q = torch.randn((d_model), requires_grad=True)
        self.proj_K = torch.randn((d_model), requires_grad=True)
        self.proj_V = torch.randn((d_model), requires_grad=True)
        self.attention = ScaledDotProductAttention(d_model)
        # self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, q, k, v, masks):
        # encoder: q = k = v = x
        # decoder: k = v = x, q = y
        # q,k,v: [batch_size, seq_len, d_model]
        # masks: [batch_size, q_len, k_len]
        batch_size = v.size(0)
        masks_extend = []
        for head in range(self.num_head):
            masks_extend.append(masks)
        masks = torch.cat(masks_extend, dim=0)
        Q = torch.mul(x, self.proj_Q.to(self.device)).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]
        K = torch.mul(x, self.proj_K.to(self.device)).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]
        V = torch.mul(x, self.proj_V.to(self.device)).permute(0, 2, 1).contiguous() # [batch_size, d_model, seq_len]

        Q = Q.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, q_len, d_head]
        K = K.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, k_len, d_head]
        V = V.view(batch_size * self.num_head, self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size * num_head, v_len, d_head]

        context = self.attention(Q, K, V, masks).permute(0, 2, 1).contiguous() # [batch_size * num_head, d_head, q_len]
        context = context.view(batch_size, self.num_head * self.d_head, -1).permute(0, 2, 1).contiguous() # [batch_size, q_len, d_model]
        out = self.dropout(context)
        out = out + q
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.act_func = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        out = self.fc1(x) # [batch_size, seq_len, hidden_size]
        out = self.act_func(out)
        out = self.fc2(out) # [batch_size, seq_len, d_model]
        out = self.act_func(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attention, position_wise_feed_forward):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = multi_head_attention
        self.position_wise_feed_forward = position_wise_feed_forward
    
    def forward(self, x, masks):
        # x: [batch_size, seq_len, d_model]
        # masks: # [batch_size, seq_len, seq_len]
        out = self.multi_head_attention(x, x, x, masks) # [batch_size, seq_len, d_model]
        out = self.position_wise_feed_forward(out) # [batch_size, seq_len, d_model]
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.embedding = Embedding(config.src_vocab_size, config.d_model, config.seq_len, config.device, config.dropout)
        self.multi_head_attention = MultiHeadAttention(config.d_model, config.num_head, config.device, config.dropout)
        self.position_wise_feed_forward = PositionWiseFeedForward(config.d_model, config.hidden_size, config.dropout)
        self.encoder_layer = EncoderLayer(
            self.multi_head_attention,
            self.position_wise_feed_forward
        )
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder_layer)
            for _ in range(config.num_encoder_layer)
        ])
        
    def forward(self, x, masks):
        # x: [batch_size, seq_len]
        # masks: [batch_size, seq_len, seq_len]
        out = self.embedding(x) # [batch_size, seq_len, d_model]
        for encoder in self.encoders:
            out = encoder(out, masks) # [batch_size, seq_len, d_model]
        return out

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, position_wise_feed_forward):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_wise_feed_forward = position_wise_feed_forward
        
    def forward(self, q, k, v, masks_yy, masks_xy):
        out = self.self_attention(q, q, q, masks_yy) # [batch_size, q_len, d_model]
        out = self.cross_attention(out, k, v, masks_xy) # [batch_size, q_len, d_model]
        out = self.position_wise_feed_forward(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.embedding = Embedding(config.tar_vocab_size, config.d_model, config.seq_len-1, config.device, config.dropout) # 因为target做了shift，所以其文本长度需要减1
        self.self_attention = MultiHeadAttention(config.d_model, config.num_head, config.dropout)
        self.cross_attention = MultiHeadAttention(config.d_model, config.num_head, config.dropout)
        self.position_wise_feed_forward = PositionWiseFeedForward(config.d_model, config.hidden_size, config.dropout)
        self.decoder_layer = DecoderLayer(
            self.self_attention,
            self.cross_attention,
            self.position_wise_feed_forward
        )
        self.decoders = nn.ModuleList([
            copy.deepcopy(self.decoder_layer)
            for _ in range(config.num_decoder_layer)
        ])

    def forward(self, encoder_out, shift_y, masks_yy, masks_xy, is_translation=False, cur_step=0):
        # encoder_out: [batch_size, x_len, d_model]
        # shift_y: [batch_size, y_len]
        # masks_yy: [batch_size, y_len, y_len]
        # masks_xy: [batch_size, y_len, x_len]
        out = self.embedding(shift_y, is_translation=is_translation, cur_step=cur_step)
        for decoder in self.decoders:
            out = decoder(out, encoder_out, encoder_out, masks_yy, masks_xy) # [batch_size, y_len, d_model]
        return out

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.device = config.device
        self.seq_len = config.seq_len

        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.linear = nn.Linear(config.d_model, config.tar_vocab_size)

    def forward(self, x, y, x_lengths, y_lengths):
        # x, y: [batch_size, seq_len]
        # x_lengths, y_lengths: [batch_size]
        masks_xx = create_atten_masks(x_lengths, x_lengths, self.seq_len, self.seq_len, self.device) # [batch_size, x_len, x_len]
        masks_yy = create_atten_masks(y_lengths, y_lengths, self.seq_len-1, self.seq_len-1, self.device) # [batch_size, y_len, y_len], y_len = config.seq_len - 1(因为target做了shift，所以其文本长度需要减1)
        masks_xy = create_atten_masks(x_lengths, y_lengths, self.seq_len, self.seq_len-1,self.device) # [batch_size, y_len, x_len]

        encoder_out = self.encoder(x, masks_xx) # [batch_size, x_len, d_model]
        decoder_out = self.decoder(encoder_out, y, masks_yy, masks_xy) # [batch_size, y_len, d_model]
        out = self.linear(decoder_out) # [batch_size, y_len, y_vocab_size]
        out = F.softmax(out, dim=-1)
        return out
    
    # 训练与infer存在gap
    # 训练时，根据上一时间步的实际词作为指导，且上一时间不的实际词还具有目标语言句子的上下文信息
    # infer时，根据上一时间部的输出作为指导（存在累计误差），且每次只输入了上一时间步的输出，不具备上下文信息
    def translate(self, x, y, x_lengths, max_x_len=32, max_y_len=32):
        # x: [batch_size, seq_len]
        # y: [batch_size, 1]
        # x_lengths: [batch_size]
        
        # 目标语言（中文）的输入只有”BOS“表示句子开始，因此y的长度为1
        # 每次都用上一个词(y)与编码器的输出预测下一个词，因此y的长度一直为1
        batch_size = y.size(0)
        y_lengths = torch.ones(batch_size).long().to(y.device)

        masks_xx = create_atten_masks(x_lengths, x_lengths, max_x_len, max_x_len, self.device) # [batch_size, x_len, x_len]
        masks_yy = create_atten_masks(y_lengths, y_lengths, 1, 1, self.device) # [batch_size, y_len, y_len]
        masks_xy = create_atten_masks(x_lengths, y_lengths, max_x_len, 1, self.device) # [batch_size, y_len, x_len]

        encoder_out = self.encoder(x, masks_xx) # [batch_size, x_len, d_model]
        predicts = []
        
        for idx in range(max_y_len):
            # 每次用上一次的输出y和编码器的输出encoder_out预测下一个词
            decoder_out = self.decoder(encoder_out, y, masks_yy, masks_xy, is_translation=True, cur_step=idx) # [batch_size, 1, d_model]
            out = self.linear(decoder_out) # [batch_size, 1, y_vocab_size]
            out = F.softmax(out, dim=-1)
            #out.max(2)[1]表示找出out第二个维度的最大值所在的位置（即预测词在词典中的index）
            y = out.max(2)[1].view(batch_size, 1) #[batch_size, 1]
            predicts.append(y)
            
        predicts = torch.cat(predicts, 1) #[batch_size, max_length]
       
        return predicts

if __name__ == "__main__":
    pass