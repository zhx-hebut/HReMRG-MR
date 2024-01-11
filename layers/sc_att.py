import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from layers.basic_att import BasicAtt
import pdb

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])
        self.attention_last3 = nn.Linear(196, 1)
        self.attention_last4 = nn.Linear(128*2, 128)

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            # channel-wise 
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)

        else:
            att_map_pool = att_map.mean(-2)

        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)
        # spatial-wise
        alpha_spatial = self.attention_last(att_map)
        alpha_spatial = alpha_spatial.squeeze(-1)

        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            spatial_value = torch.matmul(alpha_spatial, value2)
        else:
            spatial_value = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        if len(alpha_channel.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            channel_value = torch.matmul(alpha_channel, value2)
        else:
            # channel_value = torch.matmul(value2, alpha_channel.unsqueeze(-1)).squeeze(-1)
            channel_value = (value2 * alpha_channel.unsqueeze(-2)).squeeze(-1)
            channel_value = channel_value.permute(0,1,3,2)
            channel_value = self.attention_last3(channel_value).squeeze(-1)
            # channel_value = self.attention_last3(channel_value)

        value = torch.cat((spatial_value, channel_value), -1) 
        value = self.attention_last4(value)        
        # print('channel_value: ', channel_value.shape)
        # print('spatial_value: ', spatial_value.shape)
        # attn = value1 * channel_value * spatial_value
        attn = value1 * value
        return attn
