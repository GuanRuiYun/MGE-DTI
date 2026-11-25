# -*- coding: utf-8 -*-
# @Time     : 2025/4/19 上午11:15
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : layers.py
# @Project  : Graph_DTI
# @Description :

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
}


class FFN(nn.Module):
    """A two-feed-forward-layer module."""

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels * 2)
        self.lin2 = nn.Linear(channels * 2, channels)
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual
        return x


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.channels = channels
        d_k = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q, N_q = q.size(0), q.size(1)
        B_k, N_k = k.size(0), k.size(1)
        B_v, N_v = v.size(0), v.size(1)

        residual = q

        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(B_q, N_q, self.channels)
        q = self.fc(q)
        q = q + residual

        return q, attn


class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        linear_output = self.conv_linear(x)
        gate_output = torch.sigmoid(self.conv_gate(x))
        return linear_output * gate_output


class GCN(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hid_size, hid_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hid_size, out_size, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(hid_size)

    def forward(self, g, x):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, x)
        h = self.act(h)
        h1 = h
        h = self.conv2(g, h)
        h = self.act(h)
        h = self.bn1(h + h1)
        h = self.conv3(g, h)
        return h


class FAlignment(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_head, temperature=0.1):
        super(FAlignment, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.a_proj = nn.Linear(feature_dim, hidden_dim)
        self.b_proj = nn.Linear(feature_dim, hidden_dim)

        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.attention = MultiHeadAttention(num_head, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, a_feats, b_feats):
        a_proj = self.a_proj(a_feats)
        b_proj = self.b_proj(b_feats)

        cat_feats = torch.cat([a_proj, b_proj], dim=1)
        alpha = self.att_mlp(cat_feats)
        h_att = alpha * a_proj + (1 - alpha) * b_proj

        a_attn, _ = self.attention(a_proj.unsqueeze(1), b_proj.unsqueeze(1), b_proj.unsqueeze(1))
        b_attn, _ = self.attention(b_proj.unsqueeze(1), a_proj.unsqueeze(1), a_proj.unsqueeze(1))

        a_aligned = a_attn.squeeze(1)
        b_aligned = b_attn.squeeze(1)

        fused_feats = self.batch_norm(h_att + a_aligned + b_aligned)
        return a_aligned, b_aligned, fused_feats

    def contrastive_loss(self, a_feats, b_feats, mask):
        """InfoNCE alignment loss between two views of the same molecule."""
        gcn_norm = F.normalize(a_feats, dim=1)
        transformer_norm = F.normalize(b_feats, dim=1)

        sim_matrix = torch.matmul(gcn_norm, transformer_norm.T) / self.temperature
        positive_pairs = sim_matrix[mask]
        negative_pairs = sim_matrix[~mask].view(sim_matrix.size(0), -1)

        logits = torch.cat([positive_pairs.unsqueeze(1), negative_pairs], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
