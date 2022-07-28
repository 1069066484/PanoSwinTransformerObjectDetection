# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
import cv2
import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)
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


from lzx.lzx_augs.basketball_transform_torchcuda import correct_center as correct_center_cuda
from lzx.lzx_augs.basketball_transform_torchcuda import stb_adj_info as stb_adj_info_cuda
from lzx.lzx_augs.basketball_transform_torchcuda import get_v_all_patches as get_v_all_patches_cuda
from lzx.models.great_circle import great_circle22, haversine22, haversine22_approx


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # print("window_partition: x: {}  win: {}".format(torch.Size([B, H, W, C]), windows.shape))
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    # print("window_reverse: win: {}  x: {}".format(windows.shape, x.shape))
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        rel_tab_size = torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        self.relative_position_bias_table = nn.Parameter(rel_tab_size)  # 2*Wh-1 * 2*Ww-1, nH

        # self.sphere_distance = nn.Linear()
        # print("relative_position_bias_table", self.relative_position_bias_table.shape); exit()
        # relative_position_bias_table torch.Size([25, 3])

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print(coords);

        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print(coords_flatten); exit()
        #tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2],
        #        [0, 1, 2, 0, 1, 2, 0, 1, 2]])

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        # print("111 relative_position_index", relative_position_index.shape)
        # print(relative_position_index, self.relative_position_bias_table.shape)
        # 111 relative_position_index torch.Size([9, 9])
        # exit()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        num_positions = [relative_position_index.shape[0], relative_position_index.shape[1]]
        # self.sphere_position_alpha = nn.Parameter(torch.zeros(num_positions[0], num_positions[1], num_heads))
        # self.sphere_position_beta = nn.Parameter(torch.zeros(num_positions[0], num_positions[1], num_heads))

        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        self.sphere_position_alpha_table = nn.Parameter(rel_tab_size)
        self.sphere_position_beta_table = nn.Parameter(rel_tab_size)
        trunc_normal_(self.sphere_position_alpha_table, std=.02)
        trunc_normal_(self.sphere_position_beta_table, std=.02)

    def _sphere_distance(self, uv_coord=None):
        if uv_coord is None:
            return 0
        # print(uv_coord.shape) #; exit() #  torch.Size([1944, 49, 2])

        uv = uv_coord[..., :2]
        right_pos = uv_coord[..., uv_coord.shape[1] // 2, -1]

        sphere_distance = haversine22(uv, uv)

        B, H ,W = sphere_distance.shape

        sphere_position_rt = []
        for t in [lambda x: x, lambda x: x.T]:
            sphere_position_alpha = self._make_bias(self.sphere_position_alpha_table, t(self.relative_position_index))
            sphere_position_beta = self._make_bias(self.sphere_position_beta_table, t(self.relative_position_index))
            sphere_position = sphere_distance[..., None] * sphere_position_alpha[None, ...] * sphere_position_beta[None, ...]
            sphere_position_rt.append(sphere_position)
        right_pos = right_pos[:, None, None, None]
        # print(sphere_position_rt[0].shape, right_pos.shape)
        sphere_position = right_pos * sphere_position_rt[0] + (1 - right_pos) * sphere_position_rt[1]
        sphere_position = sphere_position.transpose(1,3)
        return sphere_position
        # print(u_coord); exit() # torch.Size([135, 9]) torch.Size([135, 9])

    def _make_bias(self, table, index):
        return table[index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)


    def forward(self, x, mask=None, uv_coord=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or (B, num_windows, Wh*Ww, Wh*Ww) or None
        """

        # print("333", x.shape, v_coord.shape) # torch.Size([16, 49, 96])
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self._make_bias(self.relative_position_bias_table, self.relative_position_index)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        sphere_distance = self._sphere_distance(uv_coord)
        # print(attn.shape, relative_position_bias.shape); exit() # torch.Size([135, 3, 9, 9]) torch.Size([3, 9, 9])
        if uv_coord is None:
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + sphere_distance

        if mask is not None:
            if len(mask.shape) == 3:
                # (num_windows, Wh*Ww, Wh*Ww)
                nW = mask.shape[0]
                mask = mask.unsqueeze(0)
            elif len(mask.shape) == 4:
                # (batch_size, num_windows, Wh*Ww, Wh*Ww)
                nW = mask.shape[1]
            # print("attn  mask", attn.shape, mask.shape, mask.unsqueeze(1).shape, mask.unsqueeze(1).unsqueeze(0).shape)
            # attn  mask torch.Size([16, 3, 49, 49]) torch.Size([8, 49, 49]) torch.Size([8, 1, 49, 49]) torch.Size([1, 8, 1, 49, 49])
            # print(attn.view(B_ // nW, nW, self.num_heads, N, N).shape, mask.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, win_trans_type=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.window_transition = WindowTransition(shift_size=shift_size, win_trans_type=win_trans_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def make_uv(self, v):
        u_len, v_len = v.shape[1:]

        gap = torch.abs(v[:, 1, 0] - v[:, 0, 0])

        u = torch.arange(v_len).to(v.device)[None, :].repeat(u_len, 1)

        u = u[None, ...] * gap[:, None, None]
        uv = torch.stack([u, v], -1)
        return uv

    def forward(self, x, mask_matrix, v=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # print("aaaaaaa", v.shape, x.shape)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size

        # print(1111111, x.shape, pad_l, pad_r, pad_t, pad_b, W, H)
        # print(2222222, x.shape)
        # exit()
        v = v.view(B, H, W)
        uv = self.make_uv(v)
        # print(x.shape, uv.shape); exit()
        x = torch.cat([x, uv], -1)

        # print("SwinTransformerBlock pad {}".format([]))
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        shifted_x = self.window_transition(x, reverse=False)

        uvr_c = shifted_x.shape[-1] - C

        if self.window_transition.shift_size > 0:
            attn_mask = mask_matrix
        else:
            attn_mask = None

        _, Hp, Wp, _ = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C + uvr_c * int(v is not None))  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows[..., :-uvr_c], mask=attn_mask, uv_coord=x_windows[..., -uvr_c:])  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        x = self.window_transition(shifted_x, reverse=True)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WindowTransition(nn.Module):
    def __init__(self, shift_size=0, win_trans_type=0):
        """
        @param shift_size:
        @param win_trans_type: 0-original, 1-panoTrans
        """
        super().__init__()
        self.shift_size = shift_size
        self.win_trans_type = win_trans_type
        self.ew2ns_ch = 0

    def ew2ns(self, im):
        """
        Convert a east-west panoramic representation to a north-south one
        east -> north
        west -> south
        @param im: [b, w, h, c]
        """
        if im.shape[2] % 2:
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
        if im.shape[1] % 2:
            im = F.pad(im, (0, 0, 0, 0, 0, 1))
        ms = im.shape[1] // 2
        top = im[:, :ms, :, :]
        bottom = im[:, ms:, :, :]
        top = torch.flip(top, dims=[2, 1])
        result = torch.cat([bottom, top], dim=2)
        if self.ew2ns_ch:
            result = result[..., :-1, :]
        return result

    def _get_right_u(self, x):
        u = x[..., -2]
        right_u = torch.zeros(*x.shape[1:3], dtype=u.dtype).to(u.device)
        right_u[:, :(right_u.shape[1] + 1) // 2] = 1
        x = torch.cat([x, right_u[None, :, :, None].repeat(x.shape[0], 1, 1, 1)], -1)
        return x

    def forward(self, x, reverse=False):
        if self.shift_size == 0:
            if not reverse:
                x = self._get_right_u(x)
            return x
        if self.win_trans_type == 0:
            if reverse:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        elif self.win_trans_type == 1:
            if reverse:
                x = torch.roll(x, shifts=(-self.shift_size,), dims=(1,))
                # print(111, x.shape)
                x = self.ns2we(x)
                # print(2222, x.shape)
                x = torch.roll(x, shifts=(-self.shift_size,), dims=(2,))
            else:
                # print(x[0,:,0,-2])
                x = torch.roll(x, shifts=(self.shift_size,), dims=(2,))
                # print(x.shape) # torch.Size([3, 125, 250, 98])
                x = self._get_right_u(x)
                x = self.ew2ns(x)
                # print("self.shift_size", self.shift_size)
                x = torch.roll(x, shifts=(self.shift_size,), dims=(1,))
        else:
            raise Exception("Bad win_trans_type: {}".format(self.win_trans_type))
        return x


class PanoSwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, win_trans_type=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.window_transition = WindowTransition(shift_size=shift_size, win_trans_type=win_trans_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def make_uv(self, v):
        u_len, v_len = v.shape[1:]

        gap = torch.abs(v[:, 1, 0] - v[:, 0, 0])

        u = torch.arange(v_len).to(v.device)[None, :].repeat(u_len, 1)

        u = u[None, ...] * gap[:, None, None]
        uv = torch.stack([u, v], -1)
        return uv

    def window_attention(self, shifted_x, C, mask_matrix, v_avail):
        uvr_c = shifted_x.shape[-1] - C
        attn_mask = mask_matrix if self.window_transition.shift_size else None
        _, Hp, Wp, _ = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C + uvr_c * int(v_avail))  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows[..., :-uvr_c], mask=attn_mask,
                                 uv_coord=x_windows[..., -uvr_c:])  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        return shifted_x

    def pad_x(self, x, H, W):
        pad_l = pad_t = 0
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        return x

    def forward(self, x, mask_matrix=None, v=None):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        v = v.view(B, H, W)
        uv = self.make_uv(v)
        x = torch.cat([x, uv], -1)
        if self.window_transition.win_trans_type == 0:
            # pad feature maps to multiples of window size
            x = self.pad_x(x, H, W)
            shifted_x = self.window_transition(x, reverse=False)
            shifted_x = self.window_attention(shifted_x, C, mask_matrix, v is not None)
            # reverse cyclic shift
            x = self.window_transition(shifted_x, reverse=True)
            x = x[:, :H, :W, :].contiguous()
        else:
            # print(3333, x.shape);
            shifted_x = self.window_transition(x, reverse=False)
            # cv_show1(shifted_x[0].permute(2,0,1).mean(0)[None].repeat(3,1,1))
            # print(1111, x.shape, shifted_x.shape); input()
            _, SH, SW, _ = shifted_x.shape
            uvr_c = shifted_x.shape[-1] - C
            shifted_x = self.pad_x(shifted_x, shifted_x.shape[1], shifted_x.shape[2])
            shifted_x = self.window_attention(shifted_x, C, None, v is not None)
            # reverse cyclic shift
            #################IMPORTANT
            shifted_x = shifted_x[:, :SH, :SW, :].contiguous()
            x = self.window_transition(shifted_x, reverse=True)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        win_trans_type (int): 0 - do not enable pano transition
    """

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
                 win_trans_type=0,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
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
                win_trans_type=win_trans_type,
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def _pano_update(self, img_mask, stb_adj=None):
        """
        img_mask: [1, h, w, 1]
        stb_adj: [b, 3] whether side, top, bottom is adjacent
        return: [b, h, w, 1]
        """
        b = len(stb_adj)
        updated = img_mask.repeat(b, 1, 1, 1)
        if stb_adj is None:
            return updated

        # adj[-1][0] - side adjacent
        stb_adj = [adj[-1][0] for adj in stb_adj]
        stb_adj = torch.tensor(stb_adj).bool().to(img_mask.device)

        # print(stb_adj.shape); exit()
        first_col = torch.unique(img_mask[0, 0, ...])
        last_col = torch.unique(img_mask[0, -1, ...])

        for k, v in zip(first_col, last_col):
            updated[stb_adj[:, None, None, None] & (updated == k)] = int(v)
        # print(1111, updated.shape); exit()
        return updated

    def v_downsample(self, v, H, W):
        # --downsample: x:torch.Size([3, 325, 96]) x_down:torch.Size([3, 91, 192])  v:torch.Size([3, 325])
        # print(v.shape); # torch.Size([3, 325])
        B, L = v.shape
        assert L == H * W, "input feature has wrong size"
        v = v.view(B, H, W)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if H % 2 == 1:
            v = torch.cat([v, v[:,-1:]], 1)
        if W % 2 == 1:
            v = torch.cat([v, v[:,:,-1:]], 2)
        v = F.avg_pool2d(v[:, None], (2,2))[:, 0]
        return v.view(B, -1)


    def forward(self, x, H, W, stb_adj=None, v=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # print(1111, x.shape, H, W, stb_adj, v.shape)

        # x: 2x31250x96        H: 125  W: 250
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size # Hp: 126
        Wp = int(np.ceil(W / self.window_size)) * self.window_size # Wp: 252
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
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
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = self._pano_update(img_mask, stb_adj)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        batch_size = x.shape[0]
        attn_mask = einops.rearrange(attn_mask, '(b d) p1 p2 -> b d p1 p2', b=batch_size)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask, v)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            v = self.v_downsample(v, H, W)
            return x, H, W, x_down, Wh, Ww, v
        else:
            return x, H, W, x, H, W, v


class PixelTransformer(nn.Module):
    def __init__(self, in_chans, out_dim, patch_size):
        super().__init__()
        self.in_chans = in_chans
        intermid_dim = out_dim
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.patch_size = patch_size
        intermid_dim2 = intermid_dim
        self.preprocess = nn.Sequential(
            nn.Linear(in_chans + 4, intermid_dim)
        )
        self.preprocess_nouv = nn.Sequential(
            nn.Linear(in_chans, intermid_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=intermid_dim, nhead=8, dim_feedforward=intermid_dim2, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_size[0] * self.patch_size[1], intermid_dim))

    def process_x(self, x):
        x = einops.rearrange(x, 'b c (p1 w) (p2 h) -> (b w h) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[1])
        return x

    def forward(self, x, uv_mask=None):
        shape = x.shape
        x = self.process_x(x)
        if uv_mask is None:
            x = self.preprocess_nouv(x).transpose(0, 1)
            x = x + self.pos_embedding[:,:x.shape[1]].transpose(0,1)
            x = self.encoder(x).transpose(0, 1)
            x = x.sum(1)
        else:
            uv_mask = self.process_x(uv_mask)
            pos = torch.cat([torch.cos(uv_mask[..., :2]), torch.sin(uv_mask[..., :2])], 2)
            mask = ~uv_mask[..., -1].bool()
            x = torch.cat([x, pos], 2)
            x = self.preprocess(x).transpose(0,1)
            x = self.encoder(x, src_key_padding_mask=mask).transpose(0,1)
            x = (x * mask[..., None]).sum(1) / mask.sum(1, True)
        x = einops.rearrange(x, '(b w h) s -> b s w h', w=shape[2] // self.patch_size[0], h=shape[3] // self.patch_size[1])
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, emb_conv_type='cnn', basketball_trans=True):
        super().__init__()
        assert emb_conv_type in ['cnn', 'tf']
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.basketball_trans = basketball_trans
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.emb_conv_type = emb_conv_type
        if emb_conv_type == 'cnn':
            if 0:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, embed_dim, kernel_size=patch_size, stride=patch_size)
                )
            else:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, embed_dim, kernel_size=patch_size, stride=patch_size)
                )
        else:
            self.proj = PixelTransformer(in_chans, embed_dim, patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, pano_ratio_v=None):
        """Forward function."""
        # print(pano_ratio_v)
        # padding
        B, _, H, W = x.size()

        # print(111, x.shape) # torch.Size([2, 3, 500, 1000])
        if W % self.patch_size[1] != 0:
            # print("pad W {}".format(self.patch_size[1] - W % self.patch_size[1]))
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            # print("pad H {}".format(self.patch_size[0] - H % self.patch_size[0]))
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        # print(pano_ratio_v); exit()
        if self.basketball_trans:
            assert self.patch_size[0] == self.patch_size[1]
            v_all_patches = [None] * B
            for i in range(B):
                # print(1111, i, x[i].shape)
                x[i], v_all_patches[i] = correct_center_cuda(x[i], self.patch_size[0], pano_ratio_v[i][:2], pano_ratio_v[i][2])
        else:
            v_all_patches = [get_v_all_patches_cuda(x[i], self.patch_size[0], pano_ratio_v[i][:2], pano_ratio_v[i][2])[0]
                             for i in range(B)]
        v_all_patches = torch.stack(v_all_patches)
        x = self.proj(x)  # B C W h Ww
        # print(x.shape);exit() # torch.Size([2, 96, 125, 250])
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        # print(x.shape, v_all_patches.shape); exit()
        # torch.Size([3, 96, 13, 25]) torch.Size([3, 13, 25])
        return x, v_all_patches


from lzx.utils import cv_show1


@BACKBONES.register_module()
class PanoSwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 emb_conv_type='cnn',
                 depths=[2, 2, 6, 2],
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
                 win_trans_type=0,
                 basketball_trans=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        """
        @param win_trans_type: when set it to 1, we suggest the image shape divisible by (window_size * patch_size * 4)
        """
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.win_trans_type = win_trans_type

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, emb_conv_type=emb_conv_type,
            basketball_trans=basketball_trans
        )

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

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
                win_trans_type=win_trans_type)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()
        self.coord_encoder = nn.Linear(3, embed_dim)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
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

    def _pano_ratio_v_trans(self, shape_wh, pano_ratio_v):
        def trans_single(ratio_v):
            if isinstance(ratio_v, list) and len(ratio_v) == 1:
                ratio_v = ratio_v[0]
            if isinstance(ratio_v, torch.Tensor):
                ratio_v = ratio_v.cpu().numpy()
            return [ratio_v[0], ratio_v[1], ratio_v[2]]
        if pano_ratio_v is not None:
            pano_ratio_v = [trans_single(s) for s in pano_ratio_v]
        for pano_ratio_v_i in pano_ratio_v:
            pano_ratio_v_i.append(stb_adj_info_cuda(shape_wh, self.patch_embed.patch_size[0], pano_ratio_v_i))
        # print("_pano_ratio_v_trans", pano_ratio_v)
        return pano_ratio_v

    def _pano_abs_position(self, x, pano_ratio_v):
        B = x.shape[0]
        v_all_patches = [get_v_all_patches_cuda(x[i], 1, pano_ratio_v[i][:2],
                                                pano_ratio_v[i][2], force_div=True)[0]
                         for i in range(B)]
        v_coord = torch.stack(v_all_patches, 0)
        _ , H, W = v_coord.shape

        gap = v_coord[:,1,0] - v_coord[:,0,0]
        u_coord = gap[:, None] * torch.arange(W).to(x.device)[None, :]
        u_coord = u_coord[:,None,:].repeat(1,H,1)

        # torch.Size([3, 13, 25]) torch.Size([3, 13, 25])
        # print(v_coord.shape, u_coord.shape)

        coord = torch.stack([
            torch.sin(u_coord) * torch.sin(v_coord),
            torch.cos(u_coord) * torch.sin(v_coord),
            torch.cos(v_coord),
        ], 1)
        coord = einops.rearrange(coord, "b c h w -> b (h w) c")
        coord = self.coord_encoder(coord)
        coord = einops.rearrange(coord, "b (h w) c -> b c h w", w=W, h=H)
        return coord

    def forward(self, x, pano_ratio_v=None):
        """
        Forward function.
        @param x: Tensor(b x c x h x w)
        @param pano_ratio_v: list(list(v01_start, v01_end, ori_h))
        @return:
        """
        # x = torch.rand(2,3,500,1000)
        # print(x.shape)
        # cv_show1(x)
        # print(x.shape)
        # cv_show1(x, name="1")
        # print(x[:, :, :self.patch_embed.patch_size[0]*2, :self.patch_embed.patch_size[1]*2].shape)
        # cv_show1(x[:, :, :self.patch_embed.patch_size[0], :self.patch_embed.patch_size[1]], name="2")
        pano_ratio_v = self._pano_ratio_v_trans(x.shape[2:], pano_ratio_v)

        x = x.float()
        x, v = self.patch_embed(x, pano_ratio_v)
        # print(self.patch_embed) # (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        # print(x.shape) # torch.Size([2, 96, 125, 250])

        Wh, Ww = x.size(2), x.size(3) # 125  250
        # if self.win_trans_type == 1:
        #     assert Ww == Wh * 2 and Wh % 32 == 0, "Enabling "


        # print(self.ape) # False
        if self.ape:
            if 0:
                # interpolate the position embedding to the corresponding size
                absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
                x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
                # print(self.absolute_pos_embed.shape, absolute_pos_embed.shape, absolute_pos_embed.flatten(2).transpose(1, 2).shape)
                # torch.Size([1, 96, 56, 56]) torch.Size([1, 96, 160, 152]) torch.Size([1, 24320, 96])
                # exit()
            else:
                x = (x + self._pano_abs_position(x, pano_ratio_v)).flatten(2).transpose(1, 2)
        else:
            # print(x.shape, x.flatten(2).shape, v.shape);exit()
            # torch.Size([3, 96, 13, 25]) torch.Size([3, 96, 325]) torch.Size([3, 13, 25])
            x = x.flatten(2).transpose(1, 2)
        v = v.flatten(1)
        x = self.pos_drop(x)

        # torch.Size([3, 325, 96]) torch.Size([3, 325])
        # print(x.shape, v.shape); exit()
        # there are 325 patches

        # print(x.shape, Wh, Ww)
        # torch.Size([2, 31250, 96]) 125 250
        # print(self.num_layers); exit() # 4
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww, v = layer(x, Wh, Ww, pano_ratio_v, v)
            # print(i, x_out.shape, H, W, x.shape, Wh, Ww)
            # 0 torch.Size([2, 31250, 96]) 125 250 torch.Size([2, 7875, 192]) 63 125
            # 1 torch.Size([2, 7875, 192]) 63 125 torch.Size([2, 2016, 384]) 32 63
            # 2 torch.Size([2, 2016, 384]) 32 63 torch.Size([2, 512, 768]) 16 32
            # 3 torch.Size([2, 512, 768]) 16 32 torch.Size([2, 512, 768]) 16 32

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(PanoSwinTransformer, self).train(mode)
        self._freeze_stages()


from torchvision import transforms
from lzx.utils import chceck_params_rec
def _test():
    with torch.no_grad():
        pano_swin = PanoSwinTransformer(
                     pretrain_img_size=224,
                     patch_size=4,
                     in_chans=3,
                     embed_dim=96,
                     depths=[2, 2, 6, 2],
                     num_heads=[3, 6, 12, 24],
                     window_size=7, # number of patches with a window
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.2,
                     norm_layer=nn.LayerNorm,
                     ape=True,
                     win_trans_type=1,
                     patch_norm=True,
                     basketball_trans=True,
                     out_indices=(0, 1, 2, 3),
                     frozen_stages=-1,
                     use_checkpoint=False
        )
        if 0:
            # print(pano_swin)
            chceck_params_rec(pano_swin,4)
            x = torch.rand(3,3,1000,1000*2)
            # x = torch.rand(1, 3, 12, 20)
            outs = pano_swin(x, [[0,0.8,480]] * 2 + [[0, 1.0,480]])
            # 4 [torch.Size([2, 96, 125, 250]), torch.Size([2, 192, 63, 125]), torch.Size([2, 384, 32, 63]), torch.Size([2, 768, 16, 32])]
            # 4 [(torch.Size([3, 96, 13, 25]), tensor(0.00016, grad_fn=<SumBackward0>)), (torch.Size([3, 192, 7, 13]), tensor(0.00012,
            # print(len(outs), [(o.shape, o.sum()) for o in outs])
        else:
            filename = r"E:\ori_disks\D\fduStudy\labZXD\repos\datasets\OmnidirectionalStreetViewDataset\equirectangular\JPEGImages\000006.jpg"
            im = cv2.imread(filename)
            tc = transforms.ToTensor()(im)
            # print(tc.shape) # torch.Size([3, 1000, 2000])
            tc = tc[None]
            outs = pano_swin(tc, [[0,1.0,im.shape[-1]]])



def _test_window_partition():
    x = torch.randn(2, 14, 28, 1)
    print(window_partition(x, 7).shape) # torch.Size([16, 7, 7, 1])
    print(window_partition(x, 2).shape) # torch.Size([196, 2, 2, 1])
    x2 = x[:1]
    p1 = window_partition(x2, 7)
    p2 = window_partition(x2, 2)
    print(p1.shape) # torch.Size([8, 7, 7, 1])
    print(p2.shape) # torch.Size([98, 2, 2, 1])


if __name__=='__main__':
    # _test_window_partition()
    # exit()
    _test()