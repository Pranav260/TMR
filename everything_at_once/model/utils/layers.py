import torch
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
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
            self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attention_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(Attention):
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute cross-attention
        q = q.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, num_heads, N, C // num_heads)
        k = k.unsqueeze(3).repeat(1, 1, self.num_heads, 1, 1)  # (B, N, num_heads, N, C // num_heads)
        v = v.unsqueeze(3).repeat(1, 1, self.num_heads, 1, 1)  # (B, N, num_heads, N, C // num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, N, num_heads, N, N)

        if attention_mask is not None:
            zero_attention_mask = (attention_mask == 0).view(B, 1, 1, N, 1).expand_as(attn)  # (B, N, num_heads, N, N)
            attn.masked_fill_(zero_attention_mask, -float("inf"))  # (B, N, num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).sum(dim=-2)  # (B, N, num_heads, N, C // num_heads)
        x = x.transpose(1, 2).reshape(B, N, -1)  # (B, num_heads, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FusionAttention(Attention):
    """
    Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    Copyright 2020, Ross Wightman
    """
    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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

class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=1, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None,
                 use_cls_token=False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.masking_token = nn.Parameter(torch.zeros(embed_dim))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            FusionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) # TODO: not needed, remove?
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.masking_token, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        #self.apply(_init_vit_weights)

    def forward(self, text=None, video=None, audio=None):
        # concatenate tokens
        data = [text, video, audio]
        tokens = [x['all_tokens'] for x in data if x is not None]
        tokens = torch.cat(tokens, dim=1)

        # concatenate attention masks
        tokens_mask = [x['attention_mask'] for x in data if x is not None]
        tokens_mask = torch.cat(tokens_mask, dim=1)

        # concatenate cls token
        if self.cls_token is None:
            offset = 0
        else:
            cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)
            cls_token_mask = torch.ones((1, 1)).to(tokens_mask.device).expand(tokens_mask.shape[0], -1)
            tokens_mask = torch.cat((cls_token_mask, tokens_mask), dim=1)
            offset = 1

        for block in self.blocks:
            tokens = block(tokens, attention_mask=tokens_mask)

        output = collections.OrderedDict()

        def _get_average(tokens, attention_mask):
            attention_mask = attention_mask.unsqueeze(2).expand_as(tokens)
            return (tokens * attention_mask).sum(1) / attention_mask.sum(1)

        if text is not None:
            n_tokens = text['all_tokens'].size(1)
            attention_mask = text['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            offset += n_tokens
            output['text'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if video is not None:
            n_tokens = video['all_tokens'].size(1)
            attention_mask = video['attention_mask']
            all_tokens = tokens[:, offset:offset + n_tokens]

            offset += n_tokens
            output['video'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if audio is not None:
            n_tokens = audio['all_tokens'].size(1)
            attention_mask = audio['attention_mask']
            all_tokens = tokens[:, offset: offset + n_tokens]

            offset += n_tokens
            output['audio'] = {
                "all_tokens": all_tokens,
                "attention_mask": attention_mask,
            }

        if self.cls_token is None:
            for key, value in output.items():
                output[key]['embed'] = _get_average(value["all_tokens"], value['attention_mask'])
        else:
            modalities = list(output.keys())
            modalities = '_'.join(modalities)
            if modalities not in output:
                output[modalities] = {}
            output[modalities]['embed'] = tokens[:, 0]

        return output





