
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# 自定义损失函数
# 目的：使句子中添加的padding部分不参与损失计算
class MaskCriterion(nn.Module):
    def __init__(self):
        super(MaskCriterion, self).__init__()
        
    def forward(self, predicts, targets, masks, device):
        # predicts [batch_size, max_y_len, vocab_size]
        # target [batch_size, max_y_len]
        # masks [batch_size, max_y_len]
        
        predicts = predicts.contiguous().view(-1, predicts.size(2))  # [batch_size * max_y_len, vocab_size]
        targets = targets.contiguous().view(-1, 1)   # [batch_size*max_y_len, 1]
        masks = masks.contiguous().view(-1, 1)   # [batch_size*max_y_len, 1]
        
        # predicts.gather(1, targets)为predicts[i][targets[i]]
        # 乘上masks，即只需要计算句子有效长度的预测
        # 负号：因为采用梯度下降法，所以要最大化目标词语的概率，即最小化其相反数
        loss = -predicts.gather(1, targets) * masks
        loss = torch.sum(loss) / torch.sum(masks) #平均
        
        return loss