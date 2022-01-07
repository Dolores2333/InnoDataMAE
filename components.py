# -*- coding: utf-8 _*_
# @Time : 5/1/2022 10:33 am
# @Author: ZHA Mengyue
# @FileName: components.py
# @Blog: https://github.com/Dolores2333


import torch.nn as nn
import torch.nn.functional as F

from utils import *
from einops import rearrange
from scipy.cluster.vq import kmeans2


def get_sinusoid_encoding_table(n_position, d_model):
    def get_position_angle_vector(position):
        exponent = [2 * (j // 2) / d_model for j in range(d_model)]  # [d_model,]
        position_angle_vector = position / np.power(10000, exponent)  # [d_model,]
        return position_angle_vector
    sinusoid_table = np.array([get_position_angle_vector(i) for i in range(n_position)])
    # [0::2]: 2i, [1::2]: 2i+1
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    # table of size (n_position, d_model)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def posenc(x):
    # x(bs, seq_len, z_dim)
    b , l, f = x.shape
    position_encoding = get_sinusoid_encoding_table(l, f)
    position_encoding = position_encoding.type_as(x).to(x.device).clone().detach()
    x += position_encoding
    return x


def mask_and_posenc(args, x, masks):
    """Add position encoding and Split x in to x_visible and x_masked"""
    # x(bs, seq_len, z_dim)
    b, l, f = x.shape
    position_encoding = get_sinusoid_encoding_table(args.ts_size, args.z_dim)
    position_encoding = position_encoding.type_as(x).to(x.device).clone().detach()
    x += position_encoding
    x_visible = x[~masks, :].reshape(b, -1, f)  # (bs, -1, z_dim)
    return x_visible


def posenc_and_concatenate(args, x_visible, masks):
    """Add position encoding adn concatenate x_visible and x_masked"""
    b, l, f = x_visible.shape  # (bs, -1, embed_dim)
    # print(f'Shape of x_visible is {x_visible.shape}')
    x_masked = nn.Parameter(torch.zeros(b, args.ts_size, f))[masks, :].reshape(b, -1, f)
    # print(f'Shape o x_masked id {x_masked.shape}')
    position_encoding = get_sinusoid_encoding_table(args.ts_size, f)
    position_encoding = position_encoding.expand(b, -1, -1).type_as(x_visible).to(x_visible.device).clone().detach()
    # print(f'Shape of position encoding is {position_encoding.shape}')
    # print(f'Shape masks is {masks.shape}')
    visible_position_encoding = position_encoding[~masks, :].reshape(b, -1, f)
    masked_position_encoding = position_encoding[masks, :].reshape(b, -1, f)
    x_visible += visible_position_encoding
    x_masked += masked_position_encoding
    x_full = torch.cat([x_visible, x_masked], dim=1)  # x_full(bs, seq_len, embed_dim)
    return x_full


def mask_only(x, masks):
    # x(bs, seq_len, z_dim)
    x[masks, :] = torch.normal(mean=0, std=1, size=(1,))
    return x


def calculate_average(data):
    avg_token = torch.mean(torch.mean(data, dim=0), dim=0).clone().detach()  # (z_dim, )
    return avg_token


def mask_with_average(args, x, masks):
    device = torch.device(args.device)
    avg_token = calculate_average(x)
    x[masks] = avg_token
    return x


def concatenate_only(args, x_visible, masks):
    b, l, f = x_visible.shape  # (bs, -1, hidden_dim)
    # x_masked = nn.Parameter(torch.zeros(b, args.ts_size, f))[masks, :].reshape(b, -1, f)
    x_masked = nn.Parameter(torch.normal(mean=0, std=1, size=(b, args.ts_size, f)))[masks, :].reshape(b, -1, f)
    x_full = torch.cat([x_visible, x_masked], dim=1)  # x_full (bs, seq_len, hidden_dim)
    return x_full


def reshuffle_only(args, x_visible, masks):
    b, l, f = x_visible.shape  # (bs, -1, hidden_dim)
    x_full = nn.Parameter(torch.zeros(b, args.ts_size, f))
    x_full[masks, :] = x_visible
    return x_full


"""Masked Auto Encoder Components based on RNN
    1. Encoder
    2. Decoder
    3. Quantize"""


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.z_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.hidden_dim)

    def forward(self, x):
        x_enc, _ = self.rnn(x)
        x_enc = self.fc(x_enc)
        return x_enc


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(input_size=args.hidden_dim,
                          hidden_size=args.hidden_dim,
                          num_layers=args.num_layer)
        self.fc = nn.Linear(in_features=args.hidden_dim,
                            out_features=args.z_dim)

    def forward(self, x_enc):
        x_dec, _ = self.rnn(x_enc)
        x_dec = self.fc(x_dec)
        return x_dec
