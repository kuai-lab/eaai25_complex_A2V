import math
import torch
import torch.nn as nn

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1) # [b, length, n_heads, dim_per_head]
    x = x.transpose(1, 2) # [b, n_heads, length, dim_per_head]
    x = x.reshape(bs, heads, length, -1) # [b * n_heads, length, dim_per_head]
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class AudioResampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=77,
        embedding_dim=768, # ImageBind Embedding Size
        output_dim=1024,
        ff_mult=4,
        video_length=1, # Using Frame-Wise Version or Not
    ):
        super().__init__()
        self.num_queries = num_queries 
        self.video_length = video_length

        if video_length is not None: 
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5) 
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)
        return latents
    

# ============= Audio MLP ============= #
class Mapping_Model(nn.Module):
    def __init__(self, max_length=77):
        super().__init__()
        self.max_length = max_length
        self.linear1 = torch.nn.Linear(1024,self.max_length//7*1024)
        self.linear2 = torch.nn.Linear(self.max_length//7*1024,self.max_length*1024)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(0.2)
        
    def forward(self, x):
        return self.act(self.drop(self.linear2(self.act(self.drop(self.linear1(x)))))).reshape(x.shape[0],self.max_length,1024)
   
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float =0.1, max_len : int =5, device : str = "cuda"):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len, _, _ = x.size()
        return self.encoding[:seq_len, :]  
