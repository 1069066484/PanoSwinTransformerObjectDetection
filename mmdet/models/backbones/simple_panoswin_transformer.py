"""
Implementation of PanoSwin Transformer, modified from code of Swin Transformer
    Code for CVPR23 submission "PanoSwin: a Pano-style Swin Transformer for Panorama Understanding".


Variables in this code tries to follow a consistent naming rule:
    VariableName_ChannelFormat, e.g., input_bcwh / uv_coordinates_b2wh
    ChannelFormat:
        h: height
        H: padded height
        w: width
        W: padded width
        o: window size
        O: squared window size (o * o)
        e: heads
        b: batch_size
        n: num_of_windows x batch_size
        c: channel
        C: channel + 2, where ``2'' is the channels for u and v
        t: relative table size
        T: squared relative table size (t * t)
        or any specific number if the channel size is fixed
"""

import warnings
from abc import abstractmethod
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops
from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
try:
    from ..builder import BACKBONES
except:
    from mmcv.utils import Registry
    BACKBONES = Registry('backbone')
from lzx.models.great_circle import haversine22


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """ Multilayer perceptron."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x_bhwc, window_size):
    """
    @param x_bhwc: (B, H, W, C)
    @param window_size (int): window size

    @return:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x_bhwc.shape
    x_bhowoc = x_bhwc.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows_nooc = x_bhowoc.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows_nooc


def window_reverse(windows_nooc, window_size, H, W):
    """
    @param  windows_nooc: (num_windows*B, window_size, window_size, C)
    @param window_size (int): Window size
    @param H (int): Height of image
    @param W (int): Width of image

    @return:
        x: (B, H, W, C)
            a reversed window of H x W shape
    """
    B = int(windows_nooc.shape[0] / (H * W / window_size / window_size))
    x_bhwooc = windows_nooc.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x_bhwc = x_bhwooc.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x_bhwc


def make_relative_position_index(window_size):
    """
    @param:
        window_size (int or tuple): window_size or (window_size_h, window_size_w)

    @return:
        relative_position_index_OO (window_size_h * window_size_w, window_size_h * window_size_w):
            relative position bias indices

    @demo:
        In[10]: make_relative_position_index(3)
        Out[10]:
        tensor([[12, 11, 10,  7,  6,  5,  2,  1,  0],
                [13, 12, 11,  8,  7,  6,  3,  2,  1],
                [14, 13, 12,  9,  8,  7,  4,  3,  2],
                [17, 16, 15, 12, 11, 10,  7,  6,  5],
                [18, 17, 16, 13, 12, 11,  8,  7,  6],
                [19, 18, 17, 14, 13, 12,  9,  8,  7],
                [22, 21, 20, 17, 16, 15, 12, 11, 10],
                [23, 22, 21, 18, 17, 16, 13, 12, 11],
                [24, 23, 22, 19, 18, 17, 14, 13, 12]])
    """
    window_size = to_2tuple(window_size)
    # get pair-wise relative position index for each token inside the window
    coords_h_o = torch.arange(window_size[0])
    coords_w_o = torch.arange(window_size[1])
    coords_oo = torch.stack(torch.meshgrid([coords_h_o, coords_w_o]))  # 2, Wh, Ww
    coords_flatten_2O = torch.flatten(coords_oo, 1)  # 2, Wh*Ww
    relative_coords_2OO = coords_flatten_2O[:, :, None] - coords_flatten_2O[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords_OO2 = relative_coords_2OO.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords_OO2[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords_OO2[:, :, 1] += window_size[1] - 1
    relative_coords_OO2[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index_OO = relative_coords_OO2.sum(-1)  # Wh*Ww, Wh*Ww
    return relative_position_index_OO


def make_table(window_size, num_heads):
    """
    @param: window_size (int or tuple): window_size or (window_size_h, window_size_w)
    @param: num_heads (int): number of heads in MSA

    @return:
        a two-element tuple: (sphere_position_alpha_table_Te, sphere_position_beta_table_Te)
        1. sphere_position_alpha_table_Te (table_size, number_of_heads):
            a table that gives relative position alpha bias, that is, the great-circle bias
        2. sphere_position_beta_table_Te (table_size, number_of_heads):
            a table that gives relative position alpha bias, that is, the planar bias
    """
    window_size = to_2tuple(window_size)
    rel_tab_size_Te = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
    sphere_position_alpha_table_Te = nn.Parameter(rel_tab_size_Te) # 2*Wh-1 * 2*Ww-1, nH
    sphere_position_beta_table_Te = nn.Parameter(rel_tab_size_Te) # 2*Wh-1 * 2*Ww-1, nH
    trunc_normal_(sphere_position_alpha_table_Te, std=.02)
    trunc_normal_(sphere_position_beta_table_Te, std=.02)
    return sphere_position_alpha_table_Te, sphere_position_beta_table_Te


def make_uv_hw2(H, W, device='cpu'):
    """
    @param H, W: height/width of a panorama
    @param device: 'cpu' / 'cuda:i'
    @return:
        uv_hw2 (height, width, 2):
        uv coordinates of the given panorama, u: [-pi, pi), v: [-0.5pi, 0.5pi)

    @demo:
    In[24]: make_uv_hw2(2,4)
    Out[24]:
    tensor([[[-2.3562, -0.7854],
             [-0.7854, -0.7854],
             [ 0.7854, -0.7854],
             [ 2.3562, -0.7854]],
            [[-2.3562,  0.7854],
             [-0.7854,  0.7854],
             [ 0.7854,  0.7854],
             [ 2.3562,  0.7854]]])
    """
    gap = math.pi / H
    warange = torch.arange(W).to(device)
    harange = warange[:H]

    horizon_arange, vertival_arnage = torch.meshgrid(harange, warange)
    """
    horizon_arange=                         vertival_arnage=
    tensor([[0, 0, 0, 0, 0, 0],     tensor([[0, 1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1, 1],             [0, 1, 2, 3, 4, 5],
            [2, 2, 2, 2, 2, 2]])            [0, 1, 2, 3, 4, 5]])
    """
    xy_mesh_hw2 = torch.stack([vertival_arnage, horizon_arange], -1)
    uv_hw2 = xy_mesh_hw2 * gap
    uv_hw2[..., 1] -= math.pi * 0.5
    uv_hw2[..., 0] -= math.pi
    uv_hw2 += 0.5 * gap
    return uv_hw2


class DoubleModeModule(object):
    def __init__(self, pano_mode=True):
        super().__init__()
        self.pano_mode = pano_mode
        self.set_pano_mode(pano_mode=pano_mode)

    @abstractmethod
    def set_pano_mode(self, pano_mode: bool):
        """
        The subclasses should re-implement this function to make the class members correctly set itself to the pano mode
        @param pano_mode: bool
        @return: None
        """
        self.pano_mode = pano_mode

    def switch_pano_mode(self):
        self.set_pano_mode(not self.pano_mode)


class BasicWindowAttention(nn.Module, DoubleModeModule):
    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0., proj_drop=0., pano_mode=True):
        """
        Window based multi-head self attention (W-MSA) module with relative position bias.
        It the basic class for all of shifted, non-shifted window and pitch attention window.

        @param dim (int): Number of input channels.
        @param window_size (tuple[int]): The height and width of the window.
        @param num_heads (int): Number of attention heads.
        @param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        @param attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        @param proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        @param pano_mode (bool, optional): init this module in pano mode or not. Default: True
        """
        nn.Module.__init__(self)
        DoubleModeModule.__init__(self, pano_mode=pano_mode)
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        relative_position_index_OO = make_relative_position_index(self.window_size)
        self.register_buffer("relative_position_index_OO", relative_position_index_OO)
        # self.planer_adj = nn.Parameter(torch.tensor(0.01))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.sphere_position_alpha_table_Te, self.sphere_position_beta_table_Te = make_table(self.window_size, num_heads)

    def _sphere_bias(self, uv1_nO2, uv2_nO2):
        """
        This function gives the great-circle bias in the PanoSwin
        @param uv1_nO2 (number_of_windows * batch_size, window_size_h * window_size_w, 2): uv coordinates
        @param uv2_nO2 (number_of_windows * batch_size, window_size_h * window_size_w, 2): uv coordinates
        @return:
            1. sphere_bias_neOO (number_of_windows * batch_size, number_of_heads, window_size_h * window_size_w, 2):
                the resultant great-circle bias, composed of both great-circle and planar parts
        """
        sphere_bias_beta_OOe = self._make_bias(self.sphere_position_beta_table_Te, self.relative_position_index_OO)

        if self.pano_mode:
            sphere_bias_nOO = haversine22(uv1_nO2, uv2_nO2)
            sphere_bias_alpha_OOe = self._make_bias(self.sphere_position_alpha_table_Te, self.relative_position_index_OO)
            sphere_bias_nOOe = sphere_bias_nOO[..., None] * sphere_bias_alpha_OOe[None, ...] + \
                              sphere_bias_beta_OOe
        else:
            sphere_bias_nOOe = sphere_bias_beta_OOe[None]
        sphere_bias_neOO = sphere_bias_nOOe.permute(0, 3, 1, 2)
        return sphere_bias_neOO

    def _make_bias(self, table_Te, index_OO):
        """
        @param table (table_size, number_of_heads): a bias table
        @param index (window_size_h * window_size_w, window_size_h * window_size_w): look-up indices
        @return:
            bias_OOe (window_size_h * window_size_w, window_size_h * window_size_w, number_of_heads):
                the looked up bias
        """
        bias_OOe = table_Te[index_OO.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        return bias_OOe

    def forward(self, x_windows_nOC, mask_sOO=None):
        """ Forward function.

        @param x_windows_nOC: input features with shape of (num_windows*B, N, C)
        @param mask_sOO: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or (B, num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x_windows_nOC.shape
        c = C - 2
        x_windows_nOc = x_windows_nOC[..., :c]
        uv_coord_nO2 = x_windows_nOC[..., c:]
        assert c % self.num_heads == 0, "channels should be divisible by heads, but we get channel {} and heads{}"\
            .format(c, self.num_heads)

        qkv = self.qkv(x_windows_nOc).reshape(B_, N, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn_neOO = (q @ k.transpose(-2, -1)) # e stands for heads
        sphere_bias_neOO = self._sphere_bias(uv_coord_nO2, uv_coord_nO2)

        attn_neOO = attn_neOO + sphere_bias_neOO
        if mask_sOO is not None:
            if mask_sOO.dim() == 3:
                nW = mask_sOO.shape[0]
                mask_sOO = mask_sOO.unsqueeze(0)
            elif mask_sOO.dim() == 4:
                nW = mask_sOO.shape[1]
            attn = attn_neOO.view(B_ // nW, nW, self.num_heads, N, N) + mask_sOO.unsqueeze(2)
            attn_bHWc = attn.view(-1, self.num_heads, N, N)
            attn_bHWc = self.softmax(attn_bHWc)
        else:
            attn_bHWc = self.softmax(attn_neOO)

        attn_bHWc = self.attn_drop(attn_bHWc)
        x_bOc = (attn_bHWc @ v).transpose(1, 2).reshape(B_, N, c)
        x_bOc = self.proj(x_bOc)
        x_bOc = self.proj_drop(x_bOc)
        return x_bOc



class WindowAttention(BasicWindowAttention):
    def __init__(self, qkv_bias=True, **args):
        """ Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

        @param qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        """
        super().__init__(**args)
        self.qkv = nn.Linear(args['dim'], args['dim'] * 3, bias=qkv_bias)


class WindowTransition(nn.Module, DoubleModeModule):
    def __init__(self, shift_size=0, pano_mode=False):
        """
        @param shift_size (int, optional): shift_size in shift windowing. Default: 0
        @param pano_mode (bool, optional): a default pano mode for fowwarding, True for Pano mode, or False for planar mode. Default: True
        """
        nn.Module.__init__(self)
        DoubleModeModule.__init__(self, pano_mode=pano_mode)
        self.shift_size = shift_size
        self.ew2ns_ch = 0

    def ew2ns(self, im):
        """
        Convert a east-west panoramic representation to a north-south one
        east -> north
        west -> south
        @param im: [b, w, h, c]
        """
        if im.shape[2] % 2:
            # we should ensure the width to be double of the height to enable pano-style shift windowing
            im = F.pad(im, (0, 0, 0, 1, 0, 0))
            self.ew2ns_ch = 1
        ms = im.shape[2] // 2
        left = im[..., :ms, :]
        right = im[..., ms:, :]
        right = torch.flip(right, dims=[2, 1])
        result = torch.cat([right, left], dim=1)
        return result

    def ns2we(self, im):
        """
        Convert a north-south panoramic representation to a west-east one
        @param im: [b, w, h, c]
        """
        # assert im.shape[1] == im.shape[2] * 2, "Bad shape: {}".format(im.shape)
        # assert im.shape[1] % 2 == 0
        # if im.shape[1] % 2:
        #     im = F.pad(im, (0, 0, 0, 0, 0, 1))
        assert im.shape[1] % 2 == 0
        ms = im.shape[1] // 2
        top = im[:, :ms, :, :]
        bottom = im[:, ms:, :, :]
        top = torch.flip(top, dims=[2, 1])
        result = torch.cat([bottom, top], dim=2)
        if self.ew2ns_ch:
            # if the feature map is padded when invoking 'ew2ns', we should remove it
            result = result[..., :-1, :]
            self.ew2ns_ch = 0
        return result

    def forward(self, x_bhwc, reverse=False, pano_mode=None):
        """
        Perform window transition, supporting both of swin shift and panoswin shift and the reversed
        @param x_bhwc: input feature map, (batch_size, height, width, channel)
        @param reverse: (bool, optional) reverse transition. Default: False
        @param pano_mode: (bool | None, optional) specify a pano mode, if None, we use the pano mode given in __init__. Default: None
        @return:
            x_bhwc (batch_size, height, width, channel):
                a transitioned/reversed x_bhwc
        """
        if pano_mode is None:
            pano_mode = self.pano_mode
        if pano_mode == 0:
            if reverse:
                x_bhwc = torch.roll(x_bhwc, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x_bhwc = torch.roll(x_bhwc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        elif pano_mode == 1:
            if reverse:
                x_bhwc = torch.roll(x_bhwc, shifts=(-self.shift_size,), dims=(1,))
                x_bhwc = self.ns2we(x_bhwc)
                x_bhwc = torch.roll(x_bhwc, shifts=(-self.shift_size,), dims=(2,))
            else:
                # pano-shift step1: horizontal shifting
                x_bhwc = torch.roll(x_bhwc, shifts=(self.shift_size,), dims=(2,))

                # pano-shift step2: anticlockwise rotating
                x_bhwc = self.ew2ns(x_bhwc)

                # pano-shift step3: vertical shifting
                x_bhwc = torch.roll(x_bhwc, shifts=(self.shift_size,), dims=(1,))
        else:
            raise Exception("Bad pano_mode: {}".format(pano_mode))
        return x_bhwc


class PanoSwinTransformerBlock(nn.Module, DoubleModeModule):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 pano_mode=True):
        """
        PanoSwin Transformer Block
        @param dim (int): Number of input channels.
        @param num_heads (int): Number of attention heads.
        @param window_size (int): Window size.
        @param shift_size (int): Shift size for SW-MSA.
        @param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        @param qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        @param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        @param drop (float, optional): Dropout rate. Default: 0.0
        @param attn_drop (float, optional): Attention dropout rate. Default: 0.0
        @param drop_path (float, optional): Stochastic depth rate. Default: 0.0
        @param act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        @param norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        @param pano_mode: True for Pano mode, or False for planar mdoe
        """
        nn.Module.__init__(self)

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            pano_mode=pano_mode)

        self.window_transition = WindowTransition(shift_size=shift_size, pano_mode=pano_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        DoubleModeModule.__init__(self, pano_mode=pano_mode)

    def set_pano_mode(self, pano_mode):
        self.attn.set_pano_mode(pano_mode=pano_mode)
        self.window_transition.set_pano_mode(pano_mode=pano_mode)
        self.pano_mode = pano_mode

    def window_attention(self, x_bHWC, attn_mask_sOO=None):
        attn_mask_sOO = attn_mask_sOO if self.window_transition.shift_size else None
        _, Hp, Wp, C = x_bHWC.shape
        c = C - 2
        x_windows_nooC = window_partition(x_bHWC, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_nOC = x_windows_nooC.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows_nOc = self.attn(x_windows_nOC, mask_sOO=attn_mask_sOO)  # nW*B, window_size*window_size, C
        attn_windows_noOc = attn_windows_nOc.view(-1, self.window_size, self.window_size, c)
        x_bHWc = window_reverse(attn_windows_noOc, self.window_size, Hp, Wp)  # B H' W' C
        return x_bHWc

    def pad_x(self, x, H, W):
        pad_l = pad_t = 0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        return x

    def forward(self, x_bsC, attn_mask_sOO):
        """
        @param x_bsC: shape should be [B, S, C]
        @param attn_mask_sOO: attention mask
        @return:
        """
        B, S, C = x_bsC.shape
        H, W = self.H, self.W
        assert S == H * W, "input feature has wrong size"

        shortcut_bsc = x_bsC[..., :-2]
        x_bsC = torch.cat([self.norm1(x_bsC[...,:-2]), x_bsC[...,-2:]], -1)
        uv_bsC = x_bsC[..., -2:]
        x_bhwC = x_bsC.view(B, H, W, C)
        if self.pano_mode:
            shifted_x_bhwC = self.window_transition(x_bhwC, reverse=False)
            _, SH, SW, _ = shifted_x_bhwC.shape

            # pad feature maps to multiples of window size
            shifted_x_bHWC = self.pad_x(shifted_x_bhwC, SH, SW)
            shifted_x_bHWc = self.window_attention(shifted_x_bHWC)

            # IMPORTANT: before reversing, we should first remove padding
            shifted_x_bhwc = shifted_x_bHWc[:, :SH, :SW, :].contiguous()

            # reverse pano shift
            x_bhwc = self.window_transition(shifted_x_bhwc, reverse=True)
        else:
            # pad feature maps to multiples of window size
            x_bHWC = self.pad_x(x_bhwC, H, W)
            shifted_x_bHWC = self.window_transition(x_bHWC, reverse=False)
            shifted_x_bHWc = self.window_attention(shifted_x_bHWC, attn_mask_sOO)

            # reverse cyclic shift
            x_bHWc = self.window_transition(shifted_x_bHWc, reverse=True)
            x_bhwc = x_bHWc[:, :H, :W, :].contiguous()

        x_bsc = x_bhwc.view(B, H * W, C - 2)

        # FFN
        x_bsc = shortcut_bsc + self.drop_path(x_bsc)
        x_bsc = x_bsc + self.drop_path(self.mlp(self.norm2(x_bsc)))
        x_bsC = torch.cat([x_bsc, uv_bsC], -1)
        return x_bsC


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        """ Patch Merging Layer

        @param dim (int): Number of input channels.
        @param norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x_bsc, H, W):
        """ Forward function.

        @param x_bsc: Input feature, tensor size (B, H*W, C).
        @param H, W: Spatial resolution of the input feature.
        """
        B, S, C = x_bsc.shape
        assert S == H * W, "input feature has wrong size"

        x_bhwc = x_bsc.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x_bHWc = F.pad(x_bhwc, (0, 0, 0, W % 2, 0, H % 2))
        else:
            x_bHWc = x_bhwc
        x0_bHWc = x_bHWc[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1_bHWc = x_bHWc[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2_bHWc = x_bHWc[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3_bHWc = x_bHWc[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x2_bHWc = torch.cat([x0_bHWc, x1_bHWc, x2_bHWc, x3_bHWc], -1)  # B H/2 W/2 4*C
        x2_bsc = x2_bHWc.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x2_bsc = self.norm(x2_bsc)
        x2_bsc = self.reduction(x2_bsc)
        return x2_bsc


class BasicLayer(nn.Module, DoubleModeModule):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pano_mode=True
                 ):
        """ A basic PanoSwin Transformer layer for one stage.

        @param dim (int): Number of feature channels
        @param depth (int): Depths of this stage.
        @param num_heads (int): Number of attention head.
        @param window_size (int): Local window size. Default: 7.
        @param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        @param qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        @param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        @param drop (float, optional): Dropout rate. Default: 0.0
        @param attn_drop (float, optional): Attention dropout rate. Default: 0.0
        @param drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        @param norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        @param downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        @param use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        @param pano_mode: True for Pano mode, or False for planar mdoe
        """
        nn.Module.__init__(self)
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        depth_swin = depth - depth % 2
        blocks = [
            PanoSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pano_mode=pano_mode,
            )
            for i in range(depth_swin)]

        if depth % 2:
            blocks.append(PitchAttentionModule(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                drop=drop
            ))

        # build blocks
        self.blocks = nn.ModuleList(blocks)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        DoubleModeModule.__init__(self, pano_mode=pano_mode)

    def set_pano_mode(self, pano_mode=True):
        self.pano_mode = pano_mode
        for block in self.blocks:
            block.set_pano_mode(pano_mode)

    def _get_attention_mask(self, x_bsC, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size # Hp: 126
        Wp = int(np.ceil(W / self.window_size)) * self.window_size # Wp: 252
        img_mask_1hw1 = torch.zeros((1, Hp, Wp, 1), device=x_bsC.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),  # 0, 1, 2, ..., -self.window_size
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # print(cnt, list(h), list(w))
                img_mask_1hw1[:, h, w, :] = cnt
                cnt += 1

        mask_windows_soo1 = window_partition(img_mask_1hw1, self.window_size)  # nW, window_size, window_size, 1
        mask_windows_sO = mask_windows_soo1.view(-1, self.window_size * self.window_size)
        attn_mask_sOO = mask_windows_sO.unsqueeze(1) - mask_windows_sO.unsqueeze(2)
        attn_mask_sOO = attn_mask_sOO.masked_fill(attn_mask_sOO != 0, float(-100.0)).masked_fill(attn_mask_sOO == 0, float(0.0))
        return attn_mask_sOO

    def forward(self, x_bsC, H, W):
        """ Forward function.

        @param x_bsC: Input feature, tensor size (B, H*W, C).
        @param H, W: Spatial resolution of the input feature.
        """
        device = x_bsC.device

        if self.pano_mode:
            attn_mask_sOO = None
        else:
            attn_mask_sOO = self._get_attention_mask(x_bsC, H, W)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x_bsC = checkpoint.checkpoint(blk, x_bsC, attn_mask_sOO)
            else:
                x_bsC = blk(x_bsC, attn_mask_sOO)
        x_bsc = x_bsC[...,:-2]

        if self.downsample is not None:
            x_down_bsc = self.downsample(x_bsc, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            if self.pano_mode:
                # in pano mode, we re-sample uv coordinates
                uv_down_hw2 = make_uv_hw2(Wh, Ww, x_bsc.device)
                uv_down_s2 = uv_down_hw2.view(-1, 2)
                uv_down_bs2 = uv_down_s2[None, ...].repeat(x_down_bsc.shape[0], 1, 1)
            else:
                uv_down_bs2 = torch.zeros(*x_down_bsc.shape[:2], 2, device=device)
            x_down_bsC = torch.cat([x_down_bsc, uv_down_bs2], -1)
            return x_bsc, H, W, x_down_bsC, Wh, Ww
        else:
            return x_bsc, H, W, x_bsC, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """ Image to Patch Embedding

        @param patch_size (int): Patch token size. Default: 4.
        @param in_chans (int): Number of input image channels. Default: 3.
        @param embed_dim (int): Number of linear projection output channels. Default: 96.
        @param norm_layer (nn.Module, optional): Normalization layer. Default: None
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        embed_dim_div3 = embed_dim // 3
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim_div3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim_div3),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim_div3, embed_dim_div3 * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim_div3 * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim_div3 * 2, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""

        # padding
        B, _, H, W = x.size()

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C W h Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


from lzx.utils import cv_show1


@BACKBONES.register_module()
class SimplePanoSwinTransformer(nn.Module, DoubleModeModule):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 7, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pano_mode=True
                 ):
        """ PanoSwin Transformer backbone.
            A PyTorch impl of CVPR23 submission: `PanoSwin: a Pano-style Swin Transformer for Panorama Understanding`  -

        @param patch_size (int | tuple(int)): Patch size. Default: 4.
        @param in_chans (int): Number of input image channels. Default: 3.
        @param embed_dim (int): Number of linear projection output channels. Default: 96.
        @param depths (tuple[int]): Depths of each Swin Transformer stage.
        @param num_heads (tuple[int]): Number of attention head of each stage.
        @param window_size (int): Window size. Default: 7.
        @param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        @param qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        @param qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        @param drop_rate (float): Dropout rate.
        @param attn_drop_rate (float): Attention dropout rate. Default: 0.
        @param drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        @param norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        @param ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        @param patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        @param out_indices (Sequence[int]): Output from which stages.
        @param frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters.
        @param use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        @param pano_mode (bool): True for Pano mode, or False for planar mdoe
        """
        nn.Module.__init__(self)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # absolute position embedding
        if self.ape:
            self.abs_encoder = nn.Linear(5, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pano_mode=pano_mode)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        DoubleModeModule.__init__(self, pano_mode=pano_mode)

    def set_pano_mode(self, pano_mode=True):
        self.pano_mode = pano_mode
        for layer in self.layers:
            layer.set_pano_mode(pano_mode)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        @param pretrained:(str, optional) Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _pano_abs_position(self, x_bchw):
        """
        Obtain pano-style absolute positional embeddings
        @param x_bchw: (batch_size, channel, height, width), input feature map
        @return: a two-element tuple
            1. encoded_1chw (1, channel, height, width): absolute encodings,
            2. uv_12hw (1, 2, height, width): uv coordinates, u: [-pi, pi], v: [-0.5pi, pi]
        """
        B, C, H, W = x_bchw.shape
        device = x_bchw.device

        if not self.pano_mode:
            encoded = 0
            uv_12hw = torch.zeros(1, 2, H, W, device=device, requires_grad=False)
            return encoded, uv_12hw

        uv_hw2 = make_uv_hw2(H, W, device=device)
        xyz_coord = torch.stack([
            torch.sin(uv_hw2[..., 0]) * torch.sin(uv_hw2[..., 1]),
            torch.cos(uv_hw2[..., 0]) * torch.sin(uv_hw2[..., 1]),
            torch.cos(uv_hw2[..., 1]),
        ], -1)

        xyzuv_hw5 = torch.cat([xyz_coord, uv_hw2], -1)

        encoded_1hwc = self.abs_encoder(xyzuv_hw5[None]) # xyzyx_coord
        encoded_1chw = encoded_1hwc.permute(0, 3, 1, 2)
        uv_2hw = uv_hw2.permute(2, 0, 1)
        uv_12hw = uv_2hw[None, ...]
        return encoded_1chw, uv_12hw

    def forward(self, x_bchw, pano_ratio_v=None):
        """
        Forward function.
        @param x: Tensor(b x c x h x w)
        @return:
        """
        if pano_ratio_v is not None:
            warnings.warn("Parameter pano_ratio_v for is deprecated! Please set it to None!")

        if self.pano_mode and x_bchw.shape[3] != x_bchw.shape[2] * 2:
            warnings.warn(
                "PanoSwin is configured in Pano mode, expecting channel3 == 2 * channel2, but get {} and {}, probably cause an error".
                    format(x_bchw.shape[3], x_bchw.shape[2]))


        x_bchw = x_bchw.float()
        x_bchw = self.patch_embed(x_bchw)

        Wh, Ww = x_bchw.size(2), x_bchw.size(3) # 125  250

        position_1chw, uv_1chw = self._pano_abs_position(x_bchw)

        if self.ape: x_bchw = x_bchw + position_1chw

        x_bChw = torch.cat([x_bchw, uv_1chw.repeat(x_bchw.shape[0], 1, 1, 1)], dim=1)
        x_bsC = x_bChw.flatten(2).transpose(1, 2)

        x_bsC = self.pos_drop(x_bsC)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out_bsC, H, W, x_bsC, Wh, Ww = layer(x_bsC, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out_bsC = norm_layer(x_out_bsC)
                out = x_out_bsC.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SimplePanoSwinTransformer, self).train(mode)


from lzx.pano_rotate import pano_rotate_image, pano_rotate
import math


class PitchAttentionModule(BasicWindowAttention):
    def __init__(self,
                 qkv_bias=True,
                 np_v=-0.0001,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.0,
                 mlp_ratio=4.,
                 drop=0.,
                 act_layer=nn.GELU,
                 **args):
        """
        Implementation of the pitch attention module.

        @param qkv_bias: (bool, optional)  If True, add a learnable bias to query, key, value. Default: True
        @param np_v: v_coordinate of the target north pole
        @param norm_layer: (nn.Module, optional) Normalization layer.  Default: nn.LayerNorm
        @param drop_path: (float, optional) Stochastic depth rate. Default: 0.0
        @param mlp_ratio: (float) Ratio of mlp hidden dim to embedding dim.
        @param drop: (float, optional) Dropout rate. Default: 0.0
        @param act_layer: (nn.Module, optional) Activation layer. Default: nn.GELU
        @param args: additional arguments
        """
        BasicWindowAttention.__init__(self, **args)
        mlp_hidden_dim = int(args['dim'] * mlp_ratio)
        self.mlp = Mlp(in_features=args['dim'], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(args['dim'])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(args['dim'])
        self.q_linear = nn.Linear(args['dim'], args['dim'], bias=qkv_bias)
        self.k_linear = nn.Linear(args['dim'], args['dim'], bias=qkv_bias)
        self.v_linear = nn.Linear(args['dim'], args['dim'], bias=qkv_bias)
        np_uv = torch.Tensor([1.0, np_v]) * math.pi
        self.register_buffer("np_uv", np_uv)

    @staticmethod
    def get_rotated(x_bCHW, window_size, np_uv, pad_r=0.0, pad_b=0.0):
        """
        @param x_bCHW:  (batch_size, height, width, channel) input feature map
        @param window_size: (int) window size
        @param x_bHWC: (batch_size, height, width, channel) input feature map
        @param pad_r: (float) right padding, used to correct uv coordinates
        @param pad_b: (float) bottom padding, used to correct uv coordinates
        @return:
            new_window_rotated2_bchw (batch_size, channel, height, width) :
                check PitchAttentionModule._get_rotated for detailed descrptions
        """

        # perform panoramic rotation
        rotated_bChw, _, rotated_uv_hwb = pano_rotate_image(x_bCHW, np_uv=np_uv, with_uv=True)

        B, C, H2, W2 = x_bCHW.shape

        nH2 = H2 // window_size
        nW2 = W2 // window_size

        # window viewpoint uv
        us_w = ((torch.arange(nW2).to(x_bCHW.device) * 1.0 + 0.5) / nW2 * 2.0 * (1.0 - pad_r / W2) - 1.0) * math.pi
        vs_h = ((torch.arange(nH2).to(x_bCHW.device) * 1.0 + 0.5) / nH2 * (1.0 - pad_b / H2) - 0.5) * math.pi

        v_mesh_hw, u_mesh_hw = torch.meshgrid(vs_h, us_w)
        uvs_hw2 = torch.stack([u_mesh_hw, v_mesh_hw], -1)
        uvs_n2 = einops.rearrange(uvs_hw2, "nH2 nW2 d2 -> (nH2 nW2) d2")

        # we rotate the window center to obtain a new one
        rotated_uvs_n2 = pano_rotate(np_uv, uvs_n2, reverse=False)  # (H2 W2) 2
        rotated_uvs_hw2 = einops.rearrange(rotated_uvs_n2, "(nH2 nW2) d2 -> nH2 nW2 d2", nH2=nH2, nW2=nW2)
        rotated_uvs_hw2 /= math.pi  # u: [-1,1], v[-0.5,0.5]
        rotated_uvs_hw2[..., 1] = -rotated_uvs_hw2[..., 1]

        # this step is OK, but v ranging from [-0.5, 0.5], inconsistent with y-axis arrangement of torch.Tensor image
        # so we need a flipping
        rotated_uvs_hw2 = rotated_uvs_hw2.flip(0)
        rotated_uvs_hw2[..., 1] *= 2 # u: [-1,1], v[-1,1]

        # mp1 denotes minus 1 / plus1
        # window_size_arange_mp1_o example: tensor([-0.06122, -0.04082, -0.02041,  0.00000,  0.02041,  0.04082,  0.06122])
        window_size_arange_mp1_o = (torch.arange(window_size).to(
            x_bCHW.device) + 0.5 - 0.5 * window_size) / H2  # mp1: [minus1, plus1]

        # rotated_uvs_hw2 is the window center
        # grid_bias_xy2_mp1_oo2 is the pixel uv bias of window_size_h * window_size_h size
        # rotated_uvs_hw2 + grid_bias_xy2_mp1_oo2 give uv coordinate for each pixel in the new window
        grid_bias_x2_mp1_oo, grid_bias_y2_mp1_oo = torch.meshgrid(window_size_arange_mp1_o, window_size_arange_mp1_o)
        grid_bias_xy2_mp1_oo2 = torch.stack([grid_bias_x2_mp1_oo, grid_bias_y2_mp1_oo], -1) * 2
        grid_bias_xy2_mp1_oo2[..., 0] *= 0.5
        new_window_grids_hwoo2 = rotated_uvs_hw2[:, :, None, None, :] + \
                           grid_bias_xy2_mp1_oo2[None, None, :, :, :]

        # nH2, nW2, window_size1, window_size2, d2 = new_window_grids.shape
        _, _, window_size1, window_size2, _ = new_window_grids_hwoo2.shape
        new_window_grids_sO2 = einops.rearrange(
            new_window_grids_hwoo2,
            "nH2 nW2 window_size1 window_size2 d2 -> (nH2 nW2) (window_size1 window_size2) d2")

        # we handle the situation where the uv coordinates cross the image border
        new_window_grids_sO2[new_window_grids_sO2 <= -1.0] = new_window_grids_sO2[new_window_grids_sO2 <= -1.0] + 2.0
        new_window_grids_sO2[new_window_grids_sO2 >= 1.0] = new_window_grids_sO2[new_window_grids_sO2 >= 1.0] - 2.0

        # we perform sampling
        new_window_grids_bsO2 = new_window_grids_sO2[None].repeat(B, 1, 1, 1)
        new_window_rotated_2ssO = F.grid_sample(rotated_bChw, new_window_grids_bsO2,
                                           padding_mode='border', align_corners=False)

        new_window_rotated2_bchw = einops.rearrange(
            new_window_rotated_2ssO,
            "B C (nH2 nW2) (window_size1 window_size2) -> B C (nH2 window_size1) (nW2 window_size2)",
            nW2=nW2, nH2=nH2, window_size1=window_size1, window_size2=window_size2)
        return new_window_rotated2_bchw

    def _get_rotated(self, x_bHWC, pad_r=0.0, pad_b=0.0):
        """
        Perform pitch rotation and new window sampling

        @param x_bHWC: (batch_size, height, width, channel) input feature map
        @param pad_r: (float) right padding, used to correct uv coordinates
        @param pad_b: (float) bottom padding, used to correct uv coordinates
        @return:
            1. new_window_rotated_bHWC (batch_size, height, width, channel) :
            the rotated windows, that is, height/width is divisible by window size,
            and each grid of window_size * window_size is already a sampled new window.
            For example, when window size = 7, height = 14, width = 28.
            new_window_rotated_bHWC[:,:7,:7,:] can be directly used to attend x_bHWC[:,:7,:7,:], so as
            new_window_rotated_bHWC[:,:7,7:14,:] to x_bHWC[:,:7,7:14,:] and so on.
        """
        assert self.window_size[0] == self.window_size[1]
        if not self.pano_mode:
            return x_bHWC

        x_bCHW = x_bHWC.permute(0, 3, 1, 2)
        window_size = self.window_size[0]
        new_window_rotated_bCHW = PitchAttentionModule.get_rotated(x_bCHW, window_size, self.np_uv, pad_r, pad_b)
        new_window_rotated_bHWC = new_window_rotated_bCHW.permute(0, 2, 3, 1)
        return new_window_rotated_bHWC

    def forward(self, x_bsC, mask_matrix=None):
        """ PitchAttentionModule forward function.
        It  a. accepts an panorama feature x_bsC as input,
            b. rotates its pitch and obtain rotated_x_bHWC,
            c. partition x_bsC into windows,
            d. locate the corresponding new windows in rotated_x_bHWC,
            e. perform window-wise attention between old and new windows.

        @param x_bsC: Input feature, tensor size (B, H*W, C).
        @param mask_matrix: unused.
        """
        if x_bsC.dim() == 4:
            x_bshw = x_bsC
            warnings.warn("input x dims is expected to be (B, H*W, C), but seemingly get (B, C, H, W):{}".format(x_bsC.shape))
            B, C, H, W = x_bshw.shape
            S = H * W
            x_bsC = einops.rearrange(x_bshw, "B C H W -> B (H W) C")
        else:
            assert self.H is not None and self.W is not None
            B, S, C = x_bsC.shape
            H, W = self.H, self.W

        assert S == H * W, "input feature has wrong size"
        assert self.window_size[0] == self.window_size[1]
        # assert v is not None

        window_size = self.window_size[0]

        uv_bs2 = x_bsC[..., -2:]

        shortcut_bsc = x_bsC[..., :-2]
        x_bsC[..., :-2] = self.norm1(x_bsC[..., :-2].clone())
        x_bhwC = x_bsC.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size

        x_bHWC = F.pad(x_bhwC, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # rotates its pitch and obtain rotated_x_bHWC
        if self.pano_mode:
            rotated_x_bHWC = self._get_rotated(x_bHWC, pad_r, pad_b)
        else:
            # if not pano_mode, we simply use the original x_bHWC
            rotated_x_bHWC = x_bHWC


        # partition the original feature map into windows
        x_windows_nooC = window_partition(x_bHWC, window_size)  # nW*B, window_size, window_size, C
        x_windows_nOC = x_windows_nooC.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # partition the rotated feature map into windows
        _, Hp, Wp, _ = rotated_x_bHWC.shape
        x_windows_rotated_nooC = window_partition(rotated_x_bHWC, window_size)  # nW*B, window_size, window_size, C
        x_windows_rotated_nOC = x_windows_rotated_nooC.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows_nOc = self._attention(x_windows_nOC[...,:-2],
                                       x_windows_rotated_nOC[...,:-2],
                                       x_windows_nOC[...,-2:],
                                       x_windows_rotated_nOC[...,-2:])  # nW*B, window_size*window_size, C

        attn_windows_nooc = attn_windows_nOc.view(-1, window_size, window_size, C - 2)

        x_bHWc = window_reverse(attn_windows_nooc, window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x_bhwc = x_bHWc[:, :H, :W, :].contiguous()
        else:
            x_bhwc = x_bHWc

        x_bsc = x_bhwc.view(B, H * W, C - 2)

        # FFN
        x_bsc = shortcut_bsc + self.drop_path(x_bsc)
        x_bsc = x_bsc + self.drop_path(self.mlp(self.norm2(x_bsc)))
        x_bsC = torch.cat([x_bsc, uv_bs2], -1)
        return x_bsC


    def _attention(self, x_nOc, x_rotated_nOc, uv_coord_nO2, uv_coord_rotated_nO2):
        """
        Forward function.

        @param x_nOc: (num_windows*B, window_size_h * window_size_w, C) input windows
        @param x_rotated_nOc: (num_windows*B, window_size_h * window_size_w, C) rotated windows
        @param uv_coord_nO2: (num_windows*B, window_size_h * window_size_w, 2) uv coordinates for input windows
        @param uv_coord_rotated_nO2: (num_windows*B, window_size_h * window_size_w, 2) uv coordinates for rotated windows
        @return:
            x_nOc (num_windows*B, window_size_h * window_size_w, C):
             results after MSA
        """
        B_, N, C = x_nOc.shape
        assert C % self.num_heads == 0, "channels should be divisible by heads, but we get channel {} and heads{}"\
            .format(C, self.num_heads)

        q = self.q_linear(x_nOc).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(x_rotated_nOc).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(x_nOc).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn_neOO = (q @ k.transpose(-2, -1))

        sphere_bias_neOO = self._sphere_bias(uv_coord_nO2, uv_coord_rotated_nO2)
        attn_neOO = attn_neOO + sphere_bias_neOO
        attn_neOO = self.softmax(attn_neOO)
        attn_neOO = self.attn_drop(attn_neOO)

        x_nOc = (attn_neOO @ v).transpose(1, 2).reshape(B_, N, C)
        x_nOc = self.proj(x_nOc)
        x_nOc = self.proj_drop(x_nOc)
        return x_nOc


def _test():
    swin = SimplePanoSwinTransformer(
        patch_size=4,
        in_chans=3,
        embed_dim=8,
        depths=[3, 3, 3, 2],
        num_heads=[1, 1, 1, 1],
        out_indices=(0, 1, 2, 3),
        ape=True,
        pano_mode=True
    )
    swin = swin.cuda()
    if 1:
        for s in range(196, 200):
            size2 = (s // 2, s)
            x = torch.rand(2, 3, *size2)

            outs = swin(x.cuda())
            outs[-1].sum().backward()

            print(size2, [o.shape for o in outs])
        swin.set_pano_mode(False)
    print("exit pano mode")
    for size2 in [[45,123],[44,44],[78,64],[85,39],[99,698]]:
        x = torch.rand(2, 3, *size2)
        outs = swin(x.cuda())
        print(size2, [o.shape for o in outs])


def _test_make_uv_hw2():
    H, W = 2, 4
    uv2_hw2 = make_uv_hw2(H, W)
    print(uv2_hw2.shape)
    print(uv2_hw2)


def _test_WindowTransition():
    for h in [15, 78, 32, 94, 12, 32,6 , 7, 45]:
        w = h * 2 - 1
        bhwc = torch.arange(1*h*w*1).reshape(1,h,w,1)
        trans = WindowTransition(pano_mode=True)
        # print(x)
        # print(trans(x))
        print((trans(trans(bhwc), reverse=True) == bhwc).sum(), bhwc.size(), h * w)


from fvcore.nn import FlopCountAnalysis, parameter_count_table


def _test_flop():
    with torch.no_grad():
        depths = [3, 2, 1, 0]
        pano_swin = SimplePanoSwinTransformer(
                     patch_size=4,
                     in_chans=3,
                     embed_dim=96,
                     depths=[3, 3, 7, 2][:len(depths)],
                     num_heads=[3, 6, 12, 24][:len(depths)],
                     window_size=7, # number of patches with a window
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.2,
                     norm_layer=nn.LayerNorm,
                     ape=True,
                     patch_norm=True,
                     out_indices=(0, 1, 2, 3)[:len(depths)],
                     frozen_stages=-1,
                     use_checkpoint=False
        )
        pano_swin.set_pano_mode(False)
        pano_swin.eval()
        x = torch.rand(1,3,512,1024)
        outs = pano_swin(x)
        print("sizes=", [o.shape for o in outs])
        # sizes= [torch.Size([1, 96, 128, 256]), torch.Size([1, 192, 64, 128]), torch.Size([1, 384, 32, 64]), torch.Size([1, 768, 16, 32])]
        # exit()

        # xx = pano_swin(x)
        flops = FlopCountAnalysis(pano_swin, x)
        print(flops.total()) # 1517570712
        # exit()
        # pano_ratio_v = [[0,1.0,48]]
        # flop, param = profile(pano_swin, inputs=(x, [[0,1.0,96]]))
        # pano_swin(x)
        # 1347606822
        # print(param)
        # print(flop)


from thop import profile
def _test_flop2():
    with torch.no_grad():
        depths = [3, 2, 1, 0]
        pano_swin = SimplePanoSwinTransformer(
                     patch_size=4,
                     in_chans=3,
                     embed_dim=8,
                     depths=[3, 3, 3, 2][:len(depths)],
                     num_heads=[1, 1, 1, 1][:len(depths)],
                     window_size=7, # number of patches with a window
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.2,
                     norm_layer=nn.LayerNorm,
                     ape=True,
                     patch_norm=True,
                     out_indices=(0, 1, 2, 3)[:len(depths)],
                     frozen_stages=-1,
                     use_checkpoint=False
        )
        pano_swin.set_pano_mode(True)
        pano_swin.eval()
        x = torch.rand(1,3,512,1024)
        outs = pano_swin(x)
        print("sizes=", [o.shape for o in outs])
        # sizes= [torch.Size([1, 96, 128, 256]), torch.Size([1, 192, 64, 128]), torch.Size([1, 384, 32, 64]), torch.Size([1, 768, 16, 32])]
        # exit()

        # xx = pano_swin(x)
        flops = FlopCountAnalysis(pano_swin, x)
        print(flops.total()) # 1517570712
        # exit()
        # pano_ratio_v = [[0,1.0,48]]
        flop, param = profile(pano_swin, inputs=(x,))
        # pano_swin(x)
        # 1347606822
        print(param)
        print(flop)


if __name__ == '__main__':
    _test()

