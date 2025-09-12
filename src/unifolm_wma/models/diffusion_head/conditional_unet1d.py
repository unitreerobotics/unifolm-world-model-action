import logging
import torch
import torch.nn as nn
import einops

from einops import rearrange, repeat
from typing import Union

from unifolm_wma.models.diffusion_head.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from unifolm_wma.models.diffusion_head.positional_embedding import SinusoidalPosEmb
from unifolm_wma.models.diffusion_head.base_nets import SpatialSoftmax

from unifolm_wma.utils.basics import zero_module
from unifolm_wma.utils.common import (
    checkpoint,
    exists,
    default,
)
from unifolm_wma.utils.utils import instantiate_from_config

logger = logging.getLogger(__name__)


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
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):

    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 relative_position=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

    def efficient_forward(self, x, context=None):
        spatial_self_attn = (context is None)
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        if spatial_self_attn:
            context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, t.shape[
                1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                    b * self.heads, t.shape[1], self.dim_head).contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q,
                                                      k,
                                                      v,
                                                      attn_bias=None,
                                                      op=None)
        out = (out.unsqueeze(0).reshape(
            b, self.heads, out.shape[1],
            self.dim_head).permute(0, 2, 1,
                                   3).reshape(b, out.shape[1],
                                              self.heads * self.dim_head))
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 checkpoint=True,
                 disable_self_attn=False,
                 attention_cls=None):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim,
                              context_dim=context_dim,
                              heads=n_heads,
                              dim_head=d_head,
                              dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (
            x,
        )  ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        return checkpoint(self._forward, input_tuple, self.parameters(),
                          self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x),
                       context=context if self.disable_self_attn else None,
                       mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class ActionLatentImageCrossAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 in_dim,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 use_checkpoint=True,
                 disable_self_attn=False,
                 use_linear=True):
        super().__init__()
        """
        in_channels: action input dim

        """
        self.in_channels = in_channels
        self.in_dim = in_dim
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=8,
                                       num_channels=in_channels,
                                       eps=1e-6,
                                       affine=True)

        self.proj_in_action = nn.Linear(in_dim, inner_dim)
        self.proj_in_cond = nn.Linear(context_dim, inner_dim)
        self.proj_out = zero_module(nn.Linear(inner_dim, in_dim))
        self.use_linear = use_linear

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim,
                                  disable_self_attn=disable_self_attn,
                                  checkpoint=use_checkpoint,
                                  attention_cls=attention_cls)
            for d in range(depth)
        ])

    def forward(self, x, context=None, **kwargs):
        ba, ca, da = x.shape
        b, t, c, h, w = context.shape
        context = rearrange(context, 'b t c h w -> b (t h w) c').contiguous()

        x_in = x
        x = self.norm(x)  # ba x ja x d_in
        if self.use_linear:
            x = self.proj_in_action(x)
            context = self.proj_in_cond(context)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class ConditionalResidualBlock1D(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=True,
                 use_linear_act_proj=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels,
                        out_channels,
                        kernel_size,
                        n_groups=n_groups),
        ])

        self.cond_predict_scale = cond_predict_scale
        self.use_linear_act_proj = use_linear_act_proj
        self.out_channels = out_channels
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale and use_linear_act_proj:
            cond_channels = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        B, T, _ = cond.shape

        out = self.blocks[0](x)
        if self.cond_predict_scale:
            embed = self.cond_encoder(cond)
            if self.use_linear_act_proj:
                embed = embed.reshape(B * T, -1)
                embed = embed.reshape(-1, 2, self.out_channels, 1)
            else:
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        # else:
        #     out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):

    def __init__(self,
                 input_dim,
                 n_obs_steps=1,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False,
                 horizon=16,
                 num_head_channels=64,
                 use_linear_attn=True,
                 use_linear_act_proj=True,
                 act_proj_dim=32,
                 cond_cross_attention=False,
                 context_dims=None,
                 image_size=None,
                 imagen_cond_gradient=False,
                 last_frame_only=False,
                 use_imagen_mid_only=False,
                 use_z_only=False,
                 spatial_num_kp=32,
                 obs_encoder_config=None):
        super().__init__()

        self.n_obs_steps = n_obs_steps
        self.obs_encoder = instantiate_from_config(obs_encoder_config)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + self.obs_encoder.output_shape()[-1] * self.n_obs_steps
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        local_cond_encoder = None
        down_modules = nn.ModuleList([])

        dim_a_list = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if ind == 0:
                dim_a = horizon
            else:
                dim_a = horizon // 2 * ind
            dim_a_list.append(dim_a)

            # for attention
            num_heads = dim_out // num_head_channels
            dim_head = num_head_channels
            if use_linear_act_proj:
                if use_imagen_mid_only:
                    cur_cond_dim = cond_dim + 2 * context_dims[-1]
                elif use_z_only:
                    cur_cond_dim = cond_dim + 2 * spatial_num_kp
                else:
                    cur_cond_dim = cond_dim + 2 * context_dims[ind]
            else:
                cur_cond_dim = cond_dim + horizon * context_dims[ind]

            down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cur_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        use_linear_act_proj=use_linear_act_proj),
                    ConditionalResidualBlock1D(
                        dim_out,
                        dim_out,
                        cond_dim=cur_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        use_linear_act_proj=use_linear_act_proj),
                    ActionLatentImageCrossAttention(
                        dim_out,
                        dim_a,
                        num_heads,
                        dim_head,
                        context_dim=context_dims[ind],
                        use_linear=use_linear_attn)
                    if cond_cross_attention else nn.Identity(),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cur_cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                use_linear_act_proj=use_linear_act_proj),
            ConditionalResidualBlock1D(
                mid_dim,
                mid_dim,
                cond_dim=cur_cond_dim,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                use_linear_act_proj=use_linear_act_proj),
            ActionLatentImageCrossAttention(mid_dim,
                                            dim_a_list[-1],
                                            num_heads,
                                            dim_head,
                                            context_dim=context_dims[-1],
                                            use_linear=use_linear_attn)
            if cond_cross_attention else nn.Identity(),
        ])

        up_modules = nn.ModuleList([])
        context_dims = context_dims[::-1]
        for ind, (dim_in, dim_out) in enumerate(
                reversed(in_out[1:] + [(down_dims[-1], down_dims[-1])])):
            is_last = ind >= (len(in_out) - 1)
            if use_linear_act_proj:
                if use_imagen_mid_only:
                    cur_cond_dim = cond_dim + 2 * context_dims[0]
                elif use_z_only:
                    cur_cond_dim = cond_dim + 2 * spatial_num_kp
                else:
                    cur_cond_dim = cond_dim + 2 * context_dims[ind]
            else:
                cur_cond_dim = cond_dim + horizon * context_dims[ind]
            up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out + dim_in,
                        dim_in,
                        cond_dim=cur_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        use_linear_act_proj=use_linear_act_proj),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_in,
                        cond_dim=cur_cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        use_linear_act_proj=use_linear_act_proj),
                    ActionLatentImageCrossAttention(
                        dim_in,
                        dim_a_list.pop(),
                        num_heads,
                        dim_head,
                        context_dim=context_dims[ind],
                        use_linear=use_linear_attn)
                    if cond_cross_attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        if use_z_only:
            h, w = image_size
            self.spatial_softmax_blocks = nn.ModuleList(
                [SpatialSoftmax((4, h, w), spatial_num_kp)])
        else:
            self.spatial_softmax_blocks = nn.ModuleList([])
            context_dims = context_dims[::-1]
            for ind, context_dim in enumerate(context_dims):
                h, w = image_size
                if ind != 0:
                    h //= 2**ind
                    w //= 2**ind
                net = SpatialSoftmax((context_dim, h, w), context_dim)
                self.spatial_softmax_blocks.append(net)
            self.spatial_softmax_blocks.append(net)
            self.spatial_softmax_blocks += self.spatial_softmax_blocks[
                0:4][::-1]

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        self.cond_cross_attention = cond_cross_attention
        self.use_linear_act_proj = use_linear_act_proj

        self.proj_in_action = nn.Sequential(nn.Linear(1, act_proj_dim),
                                            nn.LayerNorm(act_proj_dim))
        self.proj_in_horizon = nn.Sequential(nn.Linear(horizon, act_proj_dim),
                                             nn.LayerNorm(act_proj_dim))
        self.proj_out_action = nn.Sequential(nn.LayerNorm(act_proj_dim),
                                             nn.Linear(act_proj_dim, 1))
        self.proj_out_horizon = nn.Sequential(nn.LayerNorm(act_proj_dim),
                                              nn.Linear(act_proj_dim, horizon))
        logger.info("number of parameters: %e",
                    sum(p.numel() for p in self.parameters()))

        self.imagen_cond_gradient = imagen_cond_gradient
        self.use_imagen_mid_only = use_imagen_mid_only
        self.use_z_only = use_z_only
        self.spatial_num_kp = spatial_num_kp
        self.last_frame_only = last_frame_only
        self.horizon = horizon

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                imagen_cond=None,
                cond=None,
                **kwargs):
        """
        sample: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        imagen_cond: a list of hidden info from video gen unet
        cond: dict:
            image: (B, 3, To, h, w)
            agent_pos: (B, Ta, d)
        output: (B,T,input_dim)
        """

        if not self.imagen_cond_gradient:
            imagen_cond = [c.detach() for c in imagen_cond]

        cond = {'image': cond[0], 'agent_pos': cond[1]}

        cond['image'] = cond['image'].permute(0, 2, 1, 3,
                                              4)
        cond['image'] = rearrange(cond['image'], 'b t c h w -> (b t) c h w')
        cond['agent_pos'] = rearrange(cond['agent_pos'], 'b t d -> (b t) d')

        B, T, D = sample.shape
        if self.use_linear_act_proj:
            sample = self.proj_in_action(sample.unsqueeze(-1))
            global_cond = self.obs_encoder(cond)
            global_cond = rearrange(global_cond,
                                    '(b t) d -> b 1 (t d)',
                                    b=B,
                                    t=self.n_obs_steps)
            global_cond = repeat(global_cond,
                                 'b c d -> b (repeat c) d',
                                 repeat=T)
        else:
            sample = einops.rearrange(sample, 'b h t -> b t h')
            sample = self.proj_in_horizon(sample)
            robo_state_cond = rearrange(robo_state_cond, 'b t d -> b 1 (t d)')
            robo_state_cond = repeat(robo_state_cond,
                                     'b c d -> b (repeat c) d',
                                     repeat=2)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps],
                                     dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        (imagen_cond_down, imagen_cond_mid, imagen_cond_up
         ) = imagen_cond[0:4], imagen_cond[4], imagen_cond[5:]  #NOTE HAND CODE

        x = sample if not self.use_linear_act_proj else sample.reshape(
            B * T, D, -1)
        h = []
        for idx, modules in enumerate(self.down_modules):
            if self.cond_cross_attention:
                (resnet, resnet2, crossatten, downsample) = modules
            else:
                (resnet, resnet2, _, downsample) = modules

            # Access the cond from the unet embeds from video unet
            if self.use_imagen_mid_only:
                imagen_cond = imagen_cond_mid
            elif self.use_z_only:
                imagen_cond = kwargs['x_start'].permute(0, 2, 1, 3, 4)
            else:
                imagen_cond = imagen_cond_down[idx]
            if self.last_frame_only:
                imagen_cond = imagen_cond[:, -1].unsqueeze(1)
                imagen_cond = repeat(imagen_cond,
                                     'b t c h w -> b (repeat t) c h w',
                                     repeat=self.horizon)
            imagen_cond = rearrange(imagen_cond, 'b t c h w -> (b t) c h w')
            if self.use_imagen_mid_only:
                imagen_cond = self.spatial_softmax_blocks[len(
                    self.spatial_softmax_blocks) // 2](imagen_cond)
            elif self.use_z_only:
                imagen_cond = self.spatial_softmax_blocks[0](imagen_cond)
            else:
                imagen_cond = self.spatial_softmax_blocks[idx](imagen_cond)
            imagen_cond = rearrange(imagen_cond, '(b t) c d -> b t c d', b=B)

            if self.use_linear_act_proj:
                imagen_cond = imagen_cond.reshape(B, T, -1)
                cur_global_feature = global_feature.unsqueeze(
                    1).repeat_interleave(repeats=T, dim=1)
            else:
                imagen_cond = imagen_cond.permute(0, 3, 1, 2)
                imagen_cond = imagen_cond.reshape(B, 2, -1)
                cur_global_feature = global_feature.unsqueeze(
                    1).repeat_interleave(repeats=2, dim=1)
            cur_global_feature = torch.cat(
                [cur_global_feature, global_cond, imagen_cond], axis=-1)
            x = resnet(x, cur_global_feature)
            x = resnet2(x, cur_global_feature)
            h.append(x)
            x = downsample(x)

        #>>> mide blocks
        resnet, resnet2, _ = self.mid_modules
        # Access the cond from the unet embeds from video unet
        if self.use_z_only:
            imagen_cond = kwargs['x_start'].permute(0, 2, 1, 3, 4)
        else:
            imagen_cond = imagen_cond_mid
        if self.last_frame_only:
            imagen_cond = imagen_cond[:, -1].unsqueeze(1)
            imagen_cond = repeat(imagen_cond,
                                 'b t c h w -> b (repeat t) c h w',
                                 repeat=self.horizon)
        imagen_cond = rearrange(imagen_cond, 'b t c h w -> (b t) c h w')
        idx += 1
        if self.use_z_only:
            imagen_cond = self.spatial_softmax_blocks[0](imagen_cond)
        else:
            imagen_cond = self.spatial_softmax_blocks[idx](imagen_cond)
        imagen_cond = rearrange(imagen_cond, '(b t) c d -> b t c d', b=B)
        if self.use_linear_act_proj:
            imagen_cond = imagen_cond.reshape(B, T, -1)
            cur_global_feature = global_feature.unsqueeze(1).repeat_interleave(
                repeats=T, dim=1)
        else:
            imagen_cond = imagen_cond.permute(0, 3, 1, 2)
            imagen_cond = imagen_cond.reshape(B, 2, -1)
            cur_global_feature = global_feature.unsqueeze(1).repeat_interleave(
                repeats=2, dim=1)
        cur_global_feature = torch.cat(
            [cur_global_feature, global_cond, imagen_cond], axis=-1)
        x = resnet(x, cur_global_feature)
        x = resnet2(x, cur_global_feature)

        #>>> up blocks
        idx += 1
        for jdx, modules in enumerate(self.up_modules):
            if self.cond_cross_attention:
                (resnet, resnet2, crossatten, upsample) = modules
            else:
                (resnet, resnet2, _, upsample) = modules

            # Access the cond from the unet embeds from video unet
            if self.use_imagen_mid_only:
                imagen_cond = imagen_cond_mid
            elif self.use_z_only:
                imagen_cond = kwargs['x_start'].permute(0, 2, 1, 3, 4)
            else:
                imagen_cond = imagen_cond_up[jdx]
            if self.last_frame_only:
                imagen_cond = imagen_cond[:, -1].unsqueeze(1)
                imagen_cond = repeat(imagen_cond,
                                     'b t c h w -> b (repeat t) c h w',
                                     repeat=self.horizon)
            imagen_cond = rearrange(imagen_cond, 'b t c h w -> (b t) c h w')
            if self.use_imagen_mid_only:
                imagen_cond = self.spatial_softmax_blocks[len(
                    self.spatial_softmax_blocks) // 2](imagen_cond)
            elif self.use_z_only:
                imagen_cond = self.spatial_softmax_blocks[0](imagen_cond)
            else:
                imagen_cond = self.spatial_softmax_blocks[jdx +
                                                          idx](imagen_cond)
            imagen_cond = rearrange(imagen_cond, '(b t) c d -> b t c d', b=B)

            if self.use_linear_act_proj:
                imagen_cond = imagen_cond.reshape(B, T, -1)
                cur_global_feature = global_feature.unsqueeze(
                    1).repeat_interleave(repeats=T, dim=1)
            else:
                imagen_cond = imagen_cond.permute(0, 3, 1, 2)
                imagen_cond = imagen_cond.reshape(B, 2, -1)
                cur_global_feature = global_feature.unsqueeze(
                    1).repeat_interleave(repeats=2, dim=1)

            cur_global_feature = torch.cat(
                [cur_global_feature, global_cond, imagen_cond], axis=-1)

            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, cur_global_feature)
            x = resnet2(x, cur_global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        if self.use_linear_act_proj:
            x = x.reshape(B, T, D, -1)
            x = self.proj_out_action(x)
            x = x.reshape(B, T, D)
        else:
            x = self.proj_out_horizon(x)
            x = einops.rearrange(x, 'b t h -> b h t')
        return x
