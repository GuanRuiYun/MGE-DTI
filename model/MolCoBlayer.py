# -*- coding: utf-8 -*-
# @Time     : 2025/4/20 下午1:14
# @Author   : cqtbi
# @Email    : gryshr@163.com
# @File     : MolCoBlayer.py
# @Project  : Graph_DTI
# @Description :

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn

from .layers import FFN, MultiHeadAttention


class BGA(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super(BGA, self).__init__()
        self.in_channels = in_dim
        self.hid_channels = hid_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.act = nn.ReLU()

        self.patch_ffn = FFN(self.in_channels, dropout)
        self.patch_global_attn = MultiHeadAttention(self.num_heads, self.in_channels, dropout)
        self.patch_global_ffn = FFN(self.in_channels, dropout)
        self.patch_global_fuse = nn.Linear(self.in_channels * 2, self.in_channels)

        self.node_attention = MultiHeadAttention(self.num_heads, self.in_channels, dropout)
        self.attention = MultiHeadAttentionLayer(in_dim, hid_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(hid_dim, out_dim)
        self.O_e = nn.Linear(hid_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, g, h, e, spatial_pos, patch, patch_mask, spatial_distance=None, return_attention=False):
        h_in1 = h.clone()
        e_in1 = e.clone()

        patch = patch.long()
        patch_shape = patch.shape
        patch_flat = patch.reshape(-1, patch_shape[-1])
        patch_h = h[patch_flat]

        attn_mask = None
        if patch_mask is not None:
            if patch_mask.dim() == 4:
                attn_mask = patch_mask.reshape(-1, patch_mask.size(-2), patch_mask.size(-1))
            else:
                valid = patch_flat != patch_flat.max(dim=1, keepdim=True).values
                attn_mask = torch.matmul(valid.unsqueeze(-1).float(), valid.unsqueeze(1).float()).int()

        patch_h, _ = self.node_attention(patch_h, patch_h, patch_h, attn_mask)
        patch_h = self.patch_ffn(patch_h)

        if attn_mask is not None:
            patch_valid = (attn_mask.sum(dim=-1, keepdim=True) > 0).float()
        else:
            patch_valid = torch.ones(patch_h.size()[:2] + (1,), device=patch_h.device, dtype=patch_h.dtype)

        patch_denom = patch_valid.sum(dim=1).clamp(min=1.0)
        patch_token = (patch_h * patch_valid).sum(dim=1) / patch_denom
        patch_token = patch_token.reshape(*patch_shape[:-1], -1)
        patch_batches = patch_token if patch_token.dim() == 3 else patch_token.unsqueeze(0)

        patch_context, _ = self.patch_global_attn(patch_batches, patch_batches, patch_batches)
        patch_context = self.patch_global_ffn(patch_context)
        patch_context = patch_context.reshape(-1, patch_context.size(-1))
        patch_context = patch_context.unsqueeze(1).repeat(1, patch_h.size(1), 1)
        patch_h = F.relu(self.patch_global_fuse(torch.cat([patch_h, patch_context], dim=-1))) + patch_h
        patch_h = patch_h * patch_valid

        valid_index = patch_valid.squeeze(-1).bool()
        h[patch_flat[valid_index]] = patch_h[valid_index]

        h_attn_out, e_attn_out, graph_stats = self.attention(
            g, h, e, spatial_pos, spatial_distance=spatial_distance, return_stats=True
        )

        h = h_attn_out.view(-1, self.hid_channels)
        e = e_attn_out.view(-1, self.hid_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h
        e_in2 = e

        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        e = self.FFN_e_layer1(e)
        e = self.act(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h
            e = e_in2 + e

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        attention_stats = {
            "attention_distance_ratio": graph_stats.get("far_attention_ratio", h.sum() * 0.0),
        }
        if return_attention:
            return h, e, attention_stats
        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_heads, self.residual
        )


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_spatial_pos = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        g.apply_edges(add_spatial_pos('score', 'proj_spatial_pos'))
        g.apply_edges(add_distance_bias('score', 'spatial_distance', self.alpha))
        g.apply_edges(out_edge_features('score'))
        g.apply_edges(exp('score'))

        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e, spatial_pos, spatial_distance=None, return_stats=False):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        proj_spatial_pos = self.proj_spatial_pos(spatial_pos)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_spatial_pos'] = proj_spatial_pos.view(-1, self.num_heads, self.out_dim)

        if spatial_distance is None:
            spatial_distance = spatial_pos.detach().float().mean(dim=-1, keepdim=True)
        if spatial_distance.dim() == 1:
            spatial_distance = spatial_distance.unsqueeze(-1)
        if spatial_distance.size(-1) != 1:
            spatial_distance = spatial_distance.float().mean(dim=-1, keepdim=True)
        g.edata['spatial_distance'] = spatial_distance.detach().float()

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        e_out = g.edata['e_out']

        if return_stats:
            score = g.edata['score']
            distance_for_heads = g.edata['spatial_distance'].detach().unsqueeze(1)
            denom = score.sum().clamp(min=1e-8)
            far_attention_ratio = (score * (distance_for_heads > 2.0).float()).sum() / denom
            return h_out, e_out, {"far_attention_ratio": far_attention_ratio}

        return h_out, e_out


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func


def imp_exp_attn(implicit_attn, explicit_edge):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] + edges.data[explicit_edge])}
    return func


def add_spatial_pos(implicit_attn, spatial_pos):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] + edges.data[spatial_pos])}
    return func


def add_distance_bias(implicit_attn, spatial_distance, alpha):
    def func(edges):
        alpha_pos = torch.clamp(alpha, min=0.0)
        distance_penalty = alpha_pos * torch.relu(edges.data[spatial_distance] - 2.0).unsqueeze(-1)
        return {implicit_attn: edges.data[implicit_attn] - distance_penalty}
    return func


def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func
