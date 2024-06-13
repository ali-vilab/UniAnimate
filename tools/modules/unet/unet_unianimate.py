import math
import torch
import xformers
import xformers.ops
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper

from .util import *
# from .mha_flash import FlashAttentionBlock
from utils.registry_class import MODEL


USE_TEMPORAL_TRANSFORMER = True



class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs) + q

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention_qkv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)    

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)
    



class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class DropPath(nn.Module):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep.
    """
    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p
    
    def forward(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args
        
        # params
        x = args[0]
        b = x.size(0)
        n = (torch.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=torch.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False
        
        # drop-path index
        index = torch.where(mask)[0]
        index = index[torch.randperm(len(index))[:n]]
        if zero is not None:
            index = torch.cat([index, torch.where(zero)[0]], dim=0)
        
        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output
    
    def broadcast(self, src, dst):
        assert src.size(0) == dst.size(0)
        shape = (dst.size(0), ) + (1, ) * (dst.ndim - 1)
        return src.view(shape)




@MODEL.register_class()
class UNetSD_UniAnimate(nn.Module):

    def __init__(self,
                 config=None,
                 in_dim=4,
                 dim=512,
                 y_dim=512,
                 context_dim=1024,
                 hist_dim = 156,
                 concat_dim = 8,
                 out_dim=6,
                 dim_mult=[1, 2, 3, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=3,
                 attn_scales=[1 / 2, 1 / 4, 1 / 8],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=1,
                 temporal_attention = True,
                 use_checkpoint=False,
                 use_image_dataset=False,
                 use_fps_condition= False,
                 use_sim_mask = False,
                 misc_dropout = 0.5,
                 training=True,
                 inpainting=True,
                 p_all_zero=0.1,
                 p_all_keep=0.1,
                 zero_y = None,
                 black_image_feature = None,
                 adapter_transformer_layers = 1,
                 num_tokens=4,
                 **kwargs
                 ):
        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        super(UNetSD_UniAnimate, self).__init__()
        self.zero_y = zero_y
        self.black_image_feature = black_image_feature
        self.cfg = config
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.num_tokens = num_tokens
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        
        self.num_heads = num_heads
        
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training=training
        self.inpainting = inpainting
        self.video_compositions = self.cfg.video_compositions
        self.misc_dropout = misc_dropout
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0
        self.resolution = config.resolution
        

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        if 'image' in self.video_compositions:
            self.pre_image_condition = nn.Sequential(
                nn.Linear(self.context_dim, self.context_dim),
                nn.SiLU(),
                nn.Linear(self.context_dim, self.context_dim*self.num_tokens))

        
        if 'local_image' in self.video_compositions:
            self.local_image_embedding = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, padding=1),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((self.resolution[1]//2, self.resolution[0]//2)),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.local_image_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, dropout_atte = 0.05, mlp_dim=concat_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)
        
        if 'dwpose' in self.video_compositions:
            self.dwpose_embedding = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, padding=1),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((self.resolution[1]//2, self.resolution[0]//2)),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1))
            self.dwpose_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, dropout_atte = 0.05, mlp_dim=concat_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)
        
        if 'randomref_pose' in self.video_compositions:
            randomref_dim = 4
            self.randomref_pose2_embedding = nn.Sequential(
                    nn.Conv2d(3, concat_dim * 4, 3, padding=1),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool2d((self.resolution[1]//2, self.resolution[0]//2)),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim+randomref_dim, 3, stride=2, padding=1))
            self.randomref_pose2_embedding_after = Transformer_v2(heads=2, dim=concat_dim+randomref_dim, dim_head_k=concat_dim+randomref_dim, dim_head_v=concat_dim+randomref_dim, dropout_atte = 0.05, mlp_dim=concat_dim+randomref_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)
        
        if 'randomref' in self.video_compositions:
            randomref_dim = 4
            self.randomref_embedding2 = nn.Sequential(
                    nn.Conv2d(randomref_dim, concat_dim * 4, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(concat_dim * 4, concat_dim+randomref_dim, 3, stride=1, padding=1))
            self.randomref_embedding_after2 = Transformer_v2(heads=2, dim=concat_dim+randomref_dim, dim_head_k=concat_dim+randomref_dim, dim_head_v=concat_dim+randomref_dim, dropout_atte = 0.05, mlp_dim=concat_dim+randomref_dim, dropout_ffn = 0.05, depth=adapter_transformer_layers)
        
        ### Condition Dropout
        self.misc_dropout = DropPath(misc_dropout)

        
        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(heads = num_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        self.pre_image = nn.Sequential()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim + concat_dim, dim, 3, padding=1)])

        #### need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformer(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))
        
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                
                block = nn.ModuleList([ResBlock(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,)])
                
                if scale in attn_scales:
                    block.append(
                            SpatialTransformer(
                                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                                disable_self_attn=False, use_linear=True
                            )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(TemporalTransformer(out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
        
        # middle
        self.middle_block = nn.ModuleList([
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,),
            SpatialTransformer(
                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                disable_self_attn=False, use_linear=True
            )])        
        
        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                 TemporalTransformer( 
                            out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal,
                            multiply_zero=use_image_dataset,
                        )
                )
            else:
                self.middle_block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))        

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))
        

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                
                block = nn.ModuleList([ResBlock(in_dim + shortcut_dims.pop(), embed_dim, dropout, out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset, )])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=1024,
                            disable_self_attn=False, use_linear=True
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset
                                    )
                            )
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        
        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)
    
    def forward(self, 
        x,
        t,
        y = None,
        depth = None,
        image = None,
        motion = None,
        local_image = None,
        single_sketch = None,
        masked = None,
        canny = None,
        sketch = None,
        dwpose = None,
        randomref = None,
        histogram = None,
        fps = None,
        video_mask = None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0  # mask last frame num
        ):

        
        assert self.inpainting or masked is None, 'inpainting is not supported'

        batch, c, f, h, w= x.shape
        frames = f
        device = x.device
        self.batch = batch

        #### image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        else:
            time_rel_pos_bias = None
        

        # all-zero and all-keep masks
        zero = torch.zeros(batch, dtype=torch.bool).to(x.device)
        keep = torch.zeros(batch, dtype=torch.bool).to(x.device)
        if self.training:
            nzero = (torch.rand(batch) < self.p_all_zero).sum()
            nkeep = (torch.rand(batch) < self.p_all_keep).sum()
            index = torch.randperm(batch)
            zero[index[0:nzero]] = True
            keep[index[nzero:nzero + nkeep]] = True
        assert not (zero & keep).any()
        misc_dropout = partial(self.misc_dropout, zero = zero, keep = keep)


        concat = x.new_zeros(batch, self.concat_dim, f, h, w)
            
            
        # local_image_embedding (first frame)
        if local_image is not None:
            local_image = rearrange(local_image, 'b c f h w -> (b f) c h w')
            local_image = self.local_image_embedding(local_image)

            h = local_image.shape[2]
            local_image = self.local_image_embedding_after(rearrange(local_image, '(b f) c h w -> (b h w) f c', b = batch))
            local_image = rearrange(local_image, '(b h w) f c -> b c f h w', b = batch, h = h)

            concat = concat + misc_dropout(local_image)
        
        if dwpose is not None:
            if 'randomref_pose' in self.video_compositions:
                dwpose_random_ref = dwpose[:,:,:1].clone()
                dwpose = dwpose[:,:,1:]
            dwpose = rearrange(dwpose, 'b c f h w -> (b f) c h w')
            dwpose = self.dwpose_embedding(dwpose)

            h = dwpose.shape[2]
            dwpose = self.dwpose_embedding_after(rearrange(dwpose, '(b f) c h w -> (b h w) f c', b = batch))
            dwpose = rearrange(dwpose, '(b h w) f c -> b c f h w', b = batch, h = h)
            concat = concat + misc_dropout(dwpose)

        randomref_b = x.new_zeros(batch, self.concat_dim+4, 1, h, w)
        if randomref is not None:
            randomref = rearrange(randomref[:,:,:1,], 'b c f h w -> (b f) c h w')
            randomref = self.randomref_embedding2(randomref)

            h = randomref.shape[2]
            randomref = self.randomref_embedding_after2(rearrange(randomref, '(b f) c h w -> (b h w) f c', b = batch))
            if 'randomref_pose' in self.video_compositions:
                dwpose_random_ref = rearrange(dwpose_random_ref, 'b c f h w -> (b f) c h w')
                dwpose_random_ref = self.randomref_pose2_embedding(dwpose_random_ref)
                dwpose_random_ref = self.randomref_pose2_embedding_after(rearrange(dwpose_random_ref, '(b f) c h w -> (b h w) f c', b = batch))
                randomref = randomref + dwpose_random_ref

            randomref_a = rearrange(randomref, '(b h w) f c -> b c f h w', b = batch, h = h)
            randomref_b = randomref_b + randomref_a
            

        x = torch.cat([randomref_b, torch.cat([x, concat], dim=1)], dim=2)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.pre_image(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)

        # embeddings
        if self.use_fps_condition and fps is not None:
            e = self.time_embed(sinusoidal_embedding(t, self.dim)) + self.fps_embedding(sinusoidal_embedding(fps, self.dim))
        else:
            e = self.time_embed(sinusoidal_embedding(t, self.dim)) 
        
        context = x.new_zeros(batch, 0, self.context_dim)
        
        
        if image is not None:
            y_context = self.zero_y.repeat(batch, 1, 1)
            context = torch.cat([context, y_context], dim=1)

            image_context = misc_dropout(self.pre_image_condition(image).view(-1, self.num_tokens, self.context_dim))  # torch.cat([y[:,:-1,:], self.pre_image_condition(y[:,-1:,:]) ], dim=1) 
            context = torch.cat([context, image_context], dim=1)
        else:
            y_context = self.zero_y.repeat(batch, 1, 1)
            context = torch.cat([context, y_context], dim=1)
            image_context = torch.zeros_like(self.zero_y.repeat(batch, 1, 1))[:,:self.num_tokens]
            context = torch.cat([context, image_context], dim=1)

        # repeat f times for spatial e and context   
        e = e.repeat_interleave(repeats=f+1, dim=0) 
        context = context.repeat_interleave(repeats=f+1, dim=0) 



        ## always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)
        
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,focus_present_mask, video_mask)
        
        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,focus_present_mask, video_mask, reference=xs[-1] if len(xs) > 0 else None)
        
        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)
        return x[:,:,1:]
    
    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block,  x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference)
        else:
            x = module(x)
        return x



