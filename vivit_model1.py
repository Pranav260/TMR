import torch
from torch import nn, einsum
from torch.nn import functional as F
import torchvision as tv
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
import torchvision.models as models
from collections import OrderedDict

#Tests
# effect of softmaxing at the end on accuracy
# weired code in the  attention module what does it do with nn.Identity()
# number of head should be 12 for using all layers of the pretrained model

"""
def pretrained_pos_embedding(frames_per_clip):
    checkpoint = torch.load('vit.pth',map_location=torch.device('cpu'))
    pos_embed_weights = OrderedDict()
    for key,value in checkpoint.items():
        if key.startswith('pos_embed'):
            pos_embed_weights[key] = value
    x  = pos_embed_weights.values()
    y = next(iter(x))
    y =  repeat(y, 'v n d -> v f n d', f = frames_per_clip)
    pos_embed_weights['pos_embed'] = y
    #torch.save(pos_embed_weights, 'pos_embed.pt')
    return y

model= ViT('B_16_imagenet1k', pretrained=True)
patch_weight = model.patch_embedding.weight

def inflate_2d_filter_to_3d(filter_2d, num_frames, num_channels_in, num_channels_out, filter_height, filter_width):
    # Initialize a 3D filter with zeros
    filter_3d = torch.zeros((num_channels_out ,num_channels_in,num_frames, filter_height, filter_width))


    # Inflate the 2D filter by replicating it along the temporal dimension and averaging them
    filter_2d_1 = filter_2d.clone().detach().cpu().numpy()
    
    for i in range(num_frames):
        filter_3d[:, :, i, :, :] = torch.tensor(filter_2d_1).float()/ num_frames

    return filter_3d
"""



class TubeletEmbeddings(nn.Module):
    """
    Video to Tubelet Embedding.
    """

    def __init__(self, video_size, patch_size, num_channels=3, embed_dim=768):
        super().__init__()
        self.video_size = video_size
        self.patch_size = patch_size
        self.num_patches = (
            (video_size[2] // patch_size[2])
            * (video_size[1] // patch_size[1])
            * (video_size[0] // patch_size[0])
        )
        self.embed_dim = embed_dim

        self.projection = nn.Conv3d(
            num_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        #self.projection.weight.data = inflate_2d_filter_to_3d(patch_weight, video_size[0], num_channels, embed_dim, patch_size[1], patch_size[2])

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, num_channels, num_frames, height, width = pixel_values.shape
        x = self.projection(pixel_values) #.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    
    # dim - inner dims of embedding , inner_dim - dim of the transformer
    def __init__(self, dim, inner_dim,dropout = 0.):
        super().__init__()
        # mlp with GELU activation function
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    """
        dim: (int) - inner dimension of embeddings[default:768] 
        heads: (int) - number of attention heads[default:12] # for pretrained model
        dim_head: (int) - dimension of transformer head [default:64] 
    
    """

    def __init__(self, dim = 768, heads = 12, dim_head = 64,dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads 
        #project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # nn.Linear from 192 to (8*64)*3
        self.make_qkv = nn.Linear(dim, inner_dim *3) 

        # Linear projection to required output dimension
        self.get_output = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout))
        #if project_out else nn.Identity()
        

    def forward(self, x):

        b, n, _ = x.shape   # b=batch_size , n=197  ,where n is the input after converting the raw input and adding cls token
        h = self.heads      # h=8

        # nn.Linear from 192 to 256*3 & then split it across q,k & v each with last dimension as 256
        qkv = self.make_qkv(x).chunk(3, dim = -1)
        
        # reshaping to get the right q,k,v dimensions having 8 attn_heads(h)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # dot product of q & k after transposing k followed by a softmax layer
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale     # q.kT / sqrt(d)
        attn = dots.softmax(dim=-1)

        # dot product of attention layer with v
        output = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Final reshaping & nn.Linear to combine all attention head outputs to get final out.
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        output =  self.get_output(output)    
        # output shape = ( b, n, dim (=192) )

        return output
    
class Transformer(nn.Module):
    """ 
        dim: (int) - inner dimension of embeddings 
        depth: (int) - depth of the transformer 
        heads: (int) - number of attention heads [default:16] 
        dim_head: (int) - dimension of transformer head [default:64] 
        mlp_dim: (int) - scaling dimension for attention [default:768] 
    
    """

    def __init__(self, dim, depth, heads=8, dim_head=64, mlp_dim=3072,dropout = 0.):
        super().__init__()
        
        self.model_layers = nn.ModuleList([])
        for i in range(depth):
            self.model_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                #nn.MultiheadAttention(dim, heads,batch_first=True),
                Attention(dim, heads, dim_head,dropout=dropout),
                nn.LayerNorm(dim),
                MLP(dim, mlp_dim,dropout=dropout)
            ]))

        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):

        for layer_norm1, attention, layer_norm2, ff_net in self.model_layers:
            
            x = attention(layer_norm1(x)) + x
            x = ff_net(layer_norm2(x)) + x


        return self.layer_norm(x)



class ViViT_2(nn.Module):
    """  
    Args:
        image_size: (int) - size of input image
        patch_size: (int) - size of each patch of the image 
        num_classes: (int) - number of classes in the dataset
        frames_per_clip: (int) - number of frames in every video clip. [default:16] 
        dim: (int) - inner dimension of embeddings[default:192] 
        depth: (int) - depth of the transformer[default:4] 
        heads: (int) - number of attention heads for the transformer[default:12] 
        pooling: (str) - type of pooling[default:'mean'] 
        in_channels: (int) - number of input channels for each frame [default:3] 
        dim_head: (int) - dimension of transformer head [default:64] 
        scale_dim: (int) - scaling dimension for attention [default:4] 
    
    """

    def __init__(self, image_size, patch_size, num_classes, frames_per_clip=32, dim = 768, depth = 12, heads = 12, pooling = 'mean', in_channels = 3, dim_head = 64, scale_dim = 4,tube = False, dropout = 0.,emb_dropout = 0.):
        
        super().__init__()

        num_patches = (image_size // patch_size) ** 2   # => 196 for 224x224 images
        patch_dim = in_channels * patch_size ** 2      # => 3*16*16

        assert pooling in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # tubelet embedding
        self.tube_flag = tube
        self.tube = TubeletEmbeddings((32,3, 224,224), (2,16,16), num_channels=3, embed_dim=768)

        #patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )


        # position embeddings of shape: (1, frames_per_clip = 16, num_patches + 1 = 197, 192)
        self.pos_embed = nn.Parameter(torch.randn(1, 16, num_patches + 1, dim))


        # space (i.e. for each image) tokens of shape: (1, 1, 192). The 192 is the tokens obtained in "get_patch_emb" 
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # spatial transformer ViT
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim,dropout)

        # time dimention tokens of shape: (1, 1, 192). 
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # temporal transformer which takes in spacetransformer's output tokens as the input. 
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim,dropout)

        # pooling type, could be "mean" or "cls"
        self.dropout =  nn.Dropout(emb_dropout)
        self.pooling = pooling

        self.dim_reduce =  nn.Linear(dim, 512)

        # mlp head for final classification
        """
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            #nn.Softmax(dim=1)             #-> difference from 2 pretrained layer code
        )
        """

    def forward(self, x):

        #print(self.tubelet_emb)
        if self.tube_flag==True:
            x = x.permute(0,2,1,3,4)
            x = self.tube(x)
            #print(x)
            x =  rearrange(x, 'b c t h w -> b t (h w) c')
        else:
            x = self.to_patch_embedding(x)

        # b = batch_size , t = frames , n = number of patch embeddings= 14*14 , e = embedding size
        b, t, n, e = x.shape     # x.shape = (b, t, 196, 192) 

        # prepare cls_token for space transformers

        spatial_cls_tokens = repeat(self.spatial_token, '() n d -> b t n d', b = b, t=t)

        # concatenate cls_token to the patch embedding
        #print("prev",x.shape)
        x = torch.cat((spatial_cls_tokens, x), dim=2)     # => x shape = ( b, t, 197 ,192)
        #print(x.shape)
        # add position embedding info
        #print(x.shape)
        x += self.pos_embed[:, :, :(n + 1)]
        #print(self.pos_embed.shape)
        #print(x.shape)
        #add embedding dropout
        x = self.dropout(x)

        # club together the b & t dimension 
        x = rearrange(x, 'b t n d -> (b t) n d')

        # pass through spatial transformer
        x = self.spatial_transformer(x)

        # declub b & t dimensions
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # prepare cls_token for temporal transformers & concatenate cls_token to the patch embedding
        temporal_cls_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((temporal_cls_tokens, x), dim=1)

        # pass through spatial transformer
        x = self.temporal_transformer(x)
        #print("temporal", x.shape)
        
        # if pooling is mean, then use mean of all 197 as the output, else use output corresponding to cls token as the final x
        if self.pooling == 'mean':
            x = x.mean(dim = 1) #( b, n, dim (=192) )(1,768)
        #else:
             #x[:, 0] #( b, n, dim (=192) )

        # pass through MLP classification layer
        #print("output shape after pooling",x.shape)
        return self.dim_reduce(x) #self.classifier_head(x)
