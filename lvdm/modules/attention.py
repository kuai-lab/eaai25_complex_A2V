from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from TrailBlazer.Utils import dd_core
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import (
    zero_module,
)

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, img_cross_attention=False,
                 injection=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.image_cross_attention_scale = 1.0
        self.text_context_len = 77
        self.img_cross_attention = img_cross_attention
        if self.img_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.injection=injection

    # Temporal Attention
    def forward(self, x, context=None, mask=None, toeplitz_matric=None, context_next=None, prompt_mp_info=None, use_injection=False,
                bundle=None, bbox_per_frame=None, e_t_uc=False, uc_emb_copied=None, step=False,pos=None):
        if bundle is not None:
            strengthen_scale=bundle["trailblazer"]["temp_strengthen_scale"]
            weaken_scale=bundle["trailblazer"]["temp_weaken_scale"]

        q = self.to_q(x)
        context = default(context, x)
        
        k = self.to_k(context)
        v = self.to_v(context)

        ## Long Video
        if context_next is not None:
            all_q, all_k, all_v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
            count = torch.zeros_like(k)
            value = torch.zeros_like(k)

            def generate_weight_sequence(n):
                if n % 2 == 0:
                    max_weight = n // 2
                    weight_sequence = list(range(1, max_weight + 1, 1)) + list(range(max_weight, 0, -1))
                else:
                    max_weight = (n + 1) // 2
                    weight_sequence = list(range(1, max_weight, 1)) + [max_weight] + list(range(max_weight - 1, 0, -1))
                return weight_sequence
            
            for t_start, t_end in context_next:
                weight_sequence = generate_weight_sequence(t_end - t_start)
                weight_tensor = torch.ones_like(count[:, t_start:t_end])
                weight_tensor = weight_tensor *torch.Tensor(weight_sequence).to(x.device).unsqueeze(0).unsqueeze(-1)
                q = all_q[:, t_start:t_end]
                k = all_k[:, t_start:t_end]
                v = all_v[:, t_start:t_end]
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
                del k

                #--TrailBlazer
                if sim.shape[0] == 20480:
                    frames = len(bbox_per_frame)
                    if use_injection:
                        def temporal_doit(origin_attn):
                            temporal_attn = rearrange(origin_attn, '(h y x) f_1 f_2 -> h y x f_1 f_2', h=self.heads, y=40, x=64, f_1=16, f_2=16)
                            temporal_attn = torch.transpose(temporal_attn, 1, 2)
                            temporal_attn = dd_core(temporal_attn, 40, 64, frames, bundle, bbox_per_frame, 
                                                    strengthen_scale, weaken_scale)
                            temporal_attn = torch.transpose(temporal_attn, 1, 2)
                            temporal_attn = rearrange(temporal_attn, 'h y x f_1 f_2 -> (h y x) f_1 f_2', h=self.heads, y=40, x=64, f_1=16, f_2=16)
                            return temporal_attn
                        sim = temporal_doit(sim)
                #--

                sim = sim.softmax(dim=-1)
                out = torch.einsum('b i j, b j d -> b i d', sim, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
                del q
                value[:,t_start:t_end] += out * weight_tensor
                count[:,t_start:t_end] += weight_tensor
            out = torch.where(count>0, value/count, value)

        ## 16 Frame Video
        else:    
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            del k
            if sim.shape[0] == 20480 and e_t_uc == True:
                frames = len(bbox_per_frame)
                if use_injection:
                    def temporal_doit(origin_attn):
                        temporal_attn = rearrange(origin_attn, '(h y x) f_1 f_2 -> h y x f_1 f_2', h=self.heads, y=40, x=64, f_1=16, f_2=16)
                        temporal_attn = torch.transpose(temporal_attn, 1, 2)
                        temporal_attn = dd_core(temporal_attn, 40, 64, frames, bundle, bbox_per_frame, 
                                                strengthen_scale, weaken_scale)
                        temporal_attn = torch.transpose(temporal_attn, 1, 2)
                        temporal_attn = rearrange(temporal_attn, 'h y x f_1 f_2 -> (h y x) f_1 f_2', h=self.heads, y=40, x=64, f_1=16, f_2=16)
                        return temporal_attn
                    sim = temporal_doit(sim)
            sim = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
            del q
        return self.to_out(out)
    
    # Spatial Attention
    def efficient_forward(self, x, context=None, mask=None, toeplitz_matric=None, context_next=None, prompt_mp_info=None, use_injection=False,
                          bundle=None, bbox_per_frame=None, e_t_uc=False, uc_emb_copied=None, step=False, pos=None):
        q = self.to_q(x)
        if context is not None:
            is_context = True
        else:
            is_context = False
        context = default(context, x)
        breakpoint()
        ## considering image token additionally
        k = self.to_k(context)
        v = self.to_v(context)
        if is_context:
            uc_emb_copied_k = self.to_k(uc_emb_copied)
            uc_emb_copied_v = self.to_v(uc_emb_copied)
            uc_emb_copied_k, uc_emb_copied_v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (uc_emb_copied_k, uc_emb_copied_v))
        b, _, _ = q.shape

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v,))
        
        #---TrailBlazer
        if bundle is not None and use_injection and e_t_uc is False and is_context:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
            del k, q

            strengthen_scale=bundle["trailblazer"]["spatial_strengthen_scale"]
            weaken_scale=bundle["trailblazer"]["spatial_weaken_scale"]
            frames = len(bbox_per_frame)
            gcd = np.gcd(320, 512)
            height_multiplier = 320 / gcd
            width_multiplier = 512 / gcd
            factor = sim.shape[1] // (height_multiplier * width_multiplier)
            factor = int(np.sqrt(factor))
            dim_y = int(factor * height_multiplier)
            dim_x = int(factor * width_multiplier)

            attention_probs_4d = sim.view(
            sim.shape[0], dim_y, dim_x, sim.shape[-1]
            )
            attention_probs_4d = dd_core(attention_probs_4d, dim_x, dim_y, frames, bundle, bbox_per_frame, 
                                        strengthen_scale, weaken_scale)
            sim = attention_probs_4d.reshape(
                attention_probs_4d.shape[0], dim_y * dim_x, attention_probs_4d.shape[-1]
            )
            sim = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', sim, v)

        else:
            if is_context and e_t_uc is False and step is True:
                gcd = np.gcd(320, 512)
                height_multiplier = 320 / gcd
                width_multiplier = 512 / gcd
                factor = q.shape[1] // (height_multiplier * width_multiplier)
                factor = int(np.sqrt(factor))
                dim_y = int(factor * height_multiplier)
                dim_x = int(factor * width_multiplier)
                q = rearrange(q, 'b (h w) d -> b h w d', b=q.shape[0], h=dim_y, w=dim_x, d=q.shape[2])
                breakpoint()
                ## Left & Right----------------------------
                if pos == "LR":
                    print("position: Left&Right")
                    n = q.shape[2] // 2

                    q_1 = q[:,:,:n]
                    q_2 = q[:,:,n:]
                    q_1, q_2 = map(lambda t: rearrange(t, 'b h w d -> b (h w) d', b=q.shape[0], d=64), (q_1, q_2))
                    k_1 = uc_emb_copied_k.detach().clone()
                    k_2 = uc_emb_copied_k.detach().clone()
                    v_1 = uc_emb_copied_v.detach().clone()
                    v_2 = uc_emb_copied_v.detach().clone()

                    k_1[:,1:11] = k[:,1:11]
                    k_2[:,1:11] = k[:,11:21]
                    v_1[:,1:11] = v[:,1:11]
                    v_2[:,1:11] = v[:,11:21]
                    out1 = xformers.ops.memory_efficient_attention(q_1, k_1, v_1, attn_bias=None, op=None)
                    out2 = xformers.ops.memory_efficient_attention(q_2, k_2, v_2, attn_bias=None, op=None)
                    out1, out2 = map(lambda t: rearrange(t, 'b (h w) d -> b h w d', b=out1.shape[0], h=dim_y, d=64), (out1, out2))
                    out = torch.cat((out1,out2), dim=2)
                    out = rearrange(out, 'b h w d -> b (h w) d')

                ## Top & Down----------------------------
                elif pos=="TD":
                    print("position: Top&Down")
                    n_1 = int(q.shape[1] * 0.4)
                    n_2 = int(q.shape[1] * 0.6)
                    q_1 = q[:,:n_1]
                    q_2 = q[:,-n_2:]
                    q_1, q_2 = map(lambda t: rearrange(t, 'b h w d -> b (h w) d', b=q.shape[0], d=q.shape[3]), (q_1, q_2))

                    k_1 = uc_emb_copied_k.detach().clone()
                    k_2 = uc_emb_copied_k.detach().clone()
                    v_1 = uc_emb_copied_v.detach().clone()
                    v_2 = uc_emb_copied_v.detach().clone()

                    k_1[:,1:11] = k[:,1:11]
                    k_2[:,1:11] = k[:,11:21]
                    v_1[:,1:11] = v[:,1:11]
                    v_2[:,1:11] = v[:,11:21]
                    out1 = xformers.ops.memory_efficient_attention(q_1, k_1, v_1, attn_bias=None, op=None)
                    out2 = xformers.ops.memory_efficient_attention(q_2, k_2, v_2, attn_bias=None, op=None)

                    out1 = rearrange(out1, 'b (h w) d -> b h w d', b=out1.shape[0], h=n_1, w=dim_x, d=out1.shape[2])
                    out2 = rearrange(out2, 'b (h w) d -> b h w d', b=out1.shape[0], h=n_2, w=dim_x, d=out2.shape[2])
                    out = torch.cat((out1,out2), dim=1)
                    out = rearrange(out, 'b h w d -> b (h w) d')

                # ## Up & Down_Left & Down_Right--------------
                # n = q.shape[2] // 2
                # n_1 = int(q.shape[1] * 0.4)
                # n_2 = int(q.shape[1] * 0.6)

                # q_1 = q[:,:n_1]
                # q_2_1 = q[:,-n_2:,:n]
                # q_2_2 = q[:,-n_2:,n:]
                
                # q_1, q_2_1, q_2_2 = map(lambda t: rearrange(t, 'b h w d -> b (h w) d', b=q.shape[0], d=q.shape[3]), (q_1, q_2_1, q_2_2))

                # k_1 = uc_emb_copied_k.detach().clone()
                # k_2_1 = uc_emb_copied_k.detach().clone()
                # k_2_2 = uc_emb_copied_k.detach().clone()
                # v_1 = uc_emb_copied_v.detach().clone()
                # v_2_1 = uc_emb_copied_v.detach().clone()
                # v_2_2 = uc_emb_copied_v.detach().clone()
                # k_1[:,1:11] = k[:,1:11]
                # k_2_1[:,1:11] = k[:,11:21]
                # k_2_2[:,1:11] = k[:,21:31]
                # v_1[:,1:11] = v[:,1:11]
                # v_2_1[:,1:11] = v[:,11:21]
                # v_2_2[:,1:11] = v[:,21:31]

                # out1 = xformers.ops.memory_efficient_attention(q_1, k_1, v_1, attn_bias=None, op=None)
                # out2_1 = xformers.ops.memory_efficient_attention(q_2_1, k_2_1, v_2_1, attn_bias=None, op=None)
                # out2_2 = xformers.ops.memory_efficient_attention(q_2_2, k_2_2, v_2_2, attn_bias=None, op=None)
                # out1 = rearrange(out1, 'b (h w) d -> b h w d', b=out1.shape[0], h=n_1, w=dim_x, d=out1.shape[2])
                # out2_1 = rearrange(out2_1, 'b (h w) d -> b h w d', b=out1.shape[0], h=n_2, w=n, d=out2_1.shape[2])
                # out2_2 = rearrange(out2_2, 'b (h w) d -> b h w d', b=out1.shape[0], h=n_2, w=n, d=out2_2.shape[2])
                # out2 = torch.cat((out2_1,out2_2), dim=2)
                # out = torch.cat((out1,out2), dim=1)
                # out = rearrange(out, 'b h w d -> b (h w) d')
                ## ----------------------------------------

                ## Up_Left & Up_Right & Down_Left & Down_Right--------------
                # n = q.shape[2] // 2
                # n_1 = int(q.shape[1] * 0.4)
                # n_2 = int(q.shape[1] * 0.6)

                # q_1_1 = q[:,:n_1,:n]
                # q_1_2 = q[:,:n_1,n:]
                # q_2_1 = q[:,-n_2:,:n]
                # q_2_2 = q[:,-n_2:,n:]
                
                # q_1_1, q_1_2, q_2_1, q_2_2 = map(lambda t: rearrange(t, 'b h w d -> b (h w) d', b=q.shape[0], d=q.shape[3]), (q_1_1, q_1_2, q_2_1, q_2_2))

                # k_1_1 = uc_emb_copied_k.detach().clone()
                # k_1_2 = uc_emb_copied_k.detach().clone()
                # k_2_1 = uc_emb_copied_k.detach().clone()
                # k_2_2 = uc_emb_copied_k.detach().clone()
                # v_1_1 = uc_emb_copied_v.detach().clone()
                # v_1_2 = uc_emb_copied_v.detach().clone()
                # v_2_1 = uc_emb_copied_v.detach().clone()
                # v_2_2 = uc_emb_copied_v.detach().clone()
                # k_1_1[:,1:11] = k[:,1:11]
                # k_1_2[:,1:11] = k[:,11:21]
                # k_2_1[:,1:11] = k[:,21:31]
                # k_2_2[:,1:11] = k[:,31:41]
                # v_1_1[:,1:11] = v[:,1:11]
                # v_1_2[:,1:11] = v[:,11:21]
                # v_2_1[:,1:11] = v[:,21:31]
                # v_2_2[:,1:11] = v[:,31:41]

                # out1_1 = xformers.ops.memory_efficient_attention(q_1_1, k_1_1, v_1_1, attn_bias=None, op=None)
                # out1_2 = xformers.ops.memory_efficient_attention(q_1_2, k_1_2, v_1_2, attn_bias=None, op=None)
                # out2_1 = xformers.ops.memory_efficient_attention(q_2_1, k_2_1, v_2_1, attn_bias=None, op=None)
                # out2_2 = xformers.ops.memory_efficient_attention(q_2_2, k_2_2, v_2_2, attn_bias=None, op=None)

                # out1_1 = rearrange(out1_1, 'b (h w) d -> b h w d', b=out1_1.shape[0], h=n_1, w=n, d=out1_1.shape[2])
                # out1_2 = rearrange(out1_2, 'b (h w) d -> b h w d', b=out1_2.shape[0], h=n_1, w=n, d=out1_2.shape[2])
                # out2_1 = rearrange(out2_1, 'b (h w) d -> b h w d', b=out2_1.shape[0], h=n_2, w=n, d=out2_1.shape[2])
                # out2_2 = rearrange(out2_2, 'b (h w) d -> b h w d', b=out2_2.shape[0], h=n_2, w=n, d=out2_2.shape[2])

                # out1 = torch.cat((out1_1,out1_2), dim=2)
                # out2 = torch.cat((out2_1,out2_2), dim=2)
                # out = torch.cat((out1,out2), dim=1)
                # out = rearrange(out, 'b h w d -> b (h w) d')
                ## ----------------------------------------

            else:
                breakpoint()
                out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
 
        if exists(mask):
            raise NotImplementedError
        
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, img_cross_attention=False,
                injection=False):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None, injection=injection)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            img_cross_attention=img_cross_attention, injection=injection)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, toeplitz_matric=None, context_next=None, prompt_mp_info=None, use_injection=False,
                bundle=None, bbox_per_frame=None, e_t_uc=False, uc_emb_copied=None, step=False,
                 **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        if context is not None and mask is not None:
            input_tuple = (x, context, mask)
        ## For NeurIPS2024
        input_tuple = (x, context, mask, toeplitz_matric, context_next, prompt_mp_info, use_injection, bundle, bbox_per_frame, e_t_uc, uc_emb_copied, step)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None, toeplitz_matric=None, context_next=None, prompt_mp_info=None, use_injection=False,
                 bundle=None, bbox_per_frame=None, e_t_uc=False, uc_emb_copied=None, step=False):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask, toeplitz_matric=toeplitz_matric, context_next=context_next, prompt_mp_info=prompt_mp_info, use_injection=use_injection,
                       bundle=bundle, bbox_per_frame=bbox_per_frame, e_t_uc=e_t_uc, uc_emb_copied=uc_emb_copied, step=step) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask, toeplitz_matric=toeplitz_matric, context_next=context_next, prompt_mp_info=prompt_mp_info, use_injection=use_injection,
                       bundle=bundle, bbox_per_frame=bbox_per_frame, e_t_uc=e_t_uc, uc_emb_copied=uc_emb_copied, step=step) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, img_cross_attention=False,
                 injection=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                img_cross_attention=img_cross_attention,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                injection=injection) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear


    def forward(self, x, context=None,  **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False,
                 relative_position=False, temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            # attention_cls = None
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        if self.causal_attention:
            mask = self.mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask, **kwargs)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j, **kwargs)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        breakpoint()
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_