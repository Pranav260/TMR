import torch
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, output_dimension)
        self.cg1 = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc1(x)
        x = self.cg1(x)
        return x


class FusedGatedUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(FusedGatedUnit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class SentenceMaxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(SentenceMaxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return torch.max(x, dim=1)[0]


class FusionBlock(nn.Module):
    """
        Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        Copyright 2020, Ross Wightman
    """
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,attn_flag = False):
        super().__init__()
        self.attn_flag = attn_flag
        self.norm1 = norm_layer(dim)
        if attn_flag == False:
            self.attn = FusionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn_cross = CrossAttention_updated(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x = None,attention_mask_x = None, attention_mask_x_ = None,attention_mask = None):
        if attention_mask_x_ is None:
            x = x + self.drop_path(self.attn(self.norm1(x),attention_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn_cross(self.norm1(x),attention_mask_x,attention_mask_x_))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class CrossAttention_updated(Attention): # for depth 2 this does not work, why ??
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self,x = None,attention_mask_x=None, attention_mask_x_=None):
        x1 = x[:,:attention_mask_x.shape[1],:]
        x2 = x[:,attention_mask_x.shape[1]:,:]
        B, N, C = x1.shape
        B,N_, C_ = x2.shape

        qkv_x = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_x_ = self.qkv(x2).reshape(B, N_, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)

        q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]
        q_x_, k_x_, v_x_ = qkv_x_[0], qkv_x_[1], qkv_x_[2]

        attn_1 = (q_x @ k_x_.transpose(-2, -1)) * self.scale
        #print("attn 1 shape",attn_1.shape)
        attn_2 = (q_x_ @ k_x.transpose(-2, -1)) * self.scale
        #print("attn 2 shape",attn_2.shape)
        #print(attn.shape)
        
        #print(attention_mask_x.shape)
        if attention_mask_x is not None:
            zero_attention_mask_x = (attention_mask_x == 0).view(B,1,1,N).repeat(1,64,N_,1)
            attn_2.masked_fill_(zero_attention_mask_x, -float("inf"))
            #attention_mask_x = attention_mask_x.unsqueeze(1).unsqueeze(2)  # Expand dimensions for broadcasting
            #zero_attention_mask_x = (attention_mask_x == 0).view(B, self.num_heads,N,N_ ).expand_as(attn)  # (bs, n_heads, q_length, k_length)
            #attn.masked_fill_(zero_attention_mask_x, -float("inf"))  # (bs, n_heads, q_length, k_length)

        if attention_mask_x_ is not None:
            # Expand dimensions for broadcasting
            zero_attention_mask_x_ = (attention_mask_x_ == 0).view(B,1,1,N_).repeat(1,64,N,1)
            #print("zero attn mask.shape",zero_attention_mask_x_.shape)  # (B, 1, N_, 1)
            attn_1.masked_fill_(zero_attention_mask_x_, -float("inf"))
            #zero_attention_mask_x_ = (attention_mask_x == 0).view(B, self.num_heads,N_ ,N).expand_as(attn) # (bs, n_heads, q_length, k_length)
            #attn.masked_fill_(zero_attention_mask_x_, -float("inf"))  # (bs, n_heads, q_length, k_length)

        #print("after mas zero1",attn_1.shape)
        #print("after mas zero2",attn_2.shape)


        attn_1 = attn_1.softmax(dim=-1)
        attn_1 = self.attn_drop(attn_1)

        attn_2 = attn_2.softmax(dim=-1)
        attn_2 = self.attn_drop(attn_2)

        x1 = (attn_1 @ v_x_).transpose(1, 2).reshape(B, N, C)
        x2 = (attn_2 @ v_x).transpose(1, 2).reshape(B, N_, C)
        x = [x1,x2]
        x = torch.cat(x,dim = 1)

        #print("shape of x after attn", x.shape)
        #print("shape of x after proj",x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("after proj drop",x.shape)
        return x


class FusionAttention(Attention):
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        #print("input x ",x.shape)
        #print(self.qkv(x).shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #print("main qkv shape",qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            zero_attention_mask = (attention_mask == 0).view(B, 1, 1, N).expand_as(attn)  # (bs, n_heads, q_length, k_length)
            attn.masked_fill_(zero_attention_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("shape of proj drop in fusion attn",x.shape)
        return x

class FusionAttention_updated(Attention):
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self, x,attn_mask_t=None,attn_mask_v = None,attn_mask_a =None ):
        
        x_t = x[:,:attn_mask_t.shape[1],:]
        x_v = x[:,44:76,:]
        x_a = x[:,76:124,:]

        B, N, C = x_t.shape
        Bv,Nv,Cv = x_v.shape
        Ba,Na,Ca = x_a.shape

        qkv = self.qkv(x_t).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(x_v).reshape(Bv, Nv, 3, self.num_heads, Cv // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv3 = self.qkv(x_a).reshape(Ba, Na, 3, self.num_heads, Ca // self.num_heads).permute(2, 0, 3, 1, 4)
        #print("main qkv shape",qkv.shape)
        q, k, v = qkv[0],qkv[1], qkv[2]
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        q3, k3, v3 = qkv3[0], qkv3[1], qkv3[2]
        

        attn1 = (q @ k.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn3 = (q3 @ k3.transpose(-2, -1)) * self.scale


        if attn_mask_t is not None:
            zero_attention_mask = (attn_mask_t == 0).view(B, 1, 1, N).expand_as(attn1)  # (bs, n_heads, q_length, k_length)
            attn1.masked_fill_(zero_attention_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        if attn_mask_v is not None:
            zero_attention_mask_2 = (attn_mask_v == 0).view(Bv, 1, 1, Nv).expand_as(attn2)  # (bs, n_heads, q_length, k_length)
            attn2.masked_fill_(zero_attention_mask_2, -float("inf"))  # (bs, n_heads, q_length, k_length)

        if attn_mask_a is not None:
            zero_attention_mask_3 = (attn_mask_a == 0).view(Ba, 1, 1, Na).expand_as(attn3)  # (bs, n_heads, q_length, k_length)
            attn3.masked_fill_(zero_attention_mask_3, -float("inf"))  # (bs, n_heads, q_length, k_length)
            

        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)

        x_t = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x_v = (attn2 @ v2).transpose(1, 2).reshape(Bv, Nv, Cv)
        x_a = (attn3 @ v3).transpose(1, 2).reshape(Ba, Na, Ca)

        x = [x_t,x_v,x_a]
        x = torch.cat(x,dim = 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("shape of proj drop in fusion attn",x.shape)
        return x


def get_projection(input_dim, output_dim, projection_type):
    if projection_type == 'minimal':
        return nn.Linear(input_dim, output_dim)
    if projection_type == 'gated':
        return GatedEmbeddingUnit(input_dim, output_dim)
    elif projection_type == '':
        return nn.Identity()
    else:
        raise NotImplementedError

