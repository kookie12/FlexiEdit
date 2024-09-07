import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.fft as fft
from torchvision.utils import save_image
from einops import rearrange, repeat
import math

# Initialize Overwatch =>> Wraps `logging.Logger`
from overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

# Copyright from MasaCtrl(https://github.com/TencentARC/MasaCtrl/blob/main/masactrl/masactrl_utils.py)
class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs) # NOTE: 요기서 attention_utils.py에 있는 pipeline으로 이동!
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

# Copyright from Ti-Guided-Edit(https://github.com/Kihensarn/TI-Guided-Edit/blob/main/utils/masactrl_utils.py)
#NOTE: added by kookie 
class FE_AttentionStore(AttentionBase):
    def __init__(self, res=[64], min_step=0, max_step=1000, ref_token_idx=1, 
                 save_mask_timestep=10, threshold=0.05, save_dir=None, image_name=None):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.ref_token_idx = ref_token_idx
        self.save_mask_timestep = save_mask_timestep
        self.threshold = threshold
        self.save_dir = save_dir
        self.image_name = image_name
        self.valid_steps = 0
        self.aggregate_mask = None

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []
        
        # added by kookie
        self.save_edw_mask_timestep = 1
        self.edw_token_idx = None
        self.edw_aggregate_mask = None
        self.edw_threshold = None

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]

        # calculate cross-attention mask
        if self.cur_step == self.save_mask_timestep:
            self.aggregate_cross_attn_map(self.ref_token_idx)
            
        # added by kookie
        if self.cur_step == self.save_edw_mask_timestep:
            if self.edw_token_idx is not None:
                for idx, edx_idx in enumerate(self.edw_token_idx):
                    self.edw_aggregate_cross_attn_map(idx, edx_idx)

        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] == self.res ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def get_aggregate_mask(self):
        return self.aggregate_mask
    
    def get_edw_aggregate_mask(self):
        return self.edw_aggregate_mask
    
    # aggregate mask from cross attention map
    def aggregate_cross_attn_map(self, idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)

        mask_image = F.interpolate(image.mean(0).unsqueeze(0).unsqueeze(0), (64, 64))
        mask_image[mask_image >= self.threshold] = 1
        mask_image[mask_image < self.threshold] = 0
        mask_image = mask_image.reshape(64, 64).unsqueeze(0)
        self.aggregate_mask = mask_image

        # os.makedirs(self.save_dir, exist_ok=True)
        if self.save_dir is not None:
            save_image(mask_image, os.path.join(self.save_dir, f"mask_{self.image_name}_{self.cur_step}.jpg"))
        return image
    
    def edw_aggregate_cross_attn_map(self, idx, edx_idx):
        attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        # attn_map = self.cross_attns[2]
        
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., edx_idx]
        if isinstance(edx_idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)

        mask_image = F.interpolate(image.mean(0).unsqueeze(0).unsqueeze(0), (64, 64))
        mask_image[mask_image >= self.edw_threshold] = 1
        mask_image[mask_image < self.edw_threshold] = 0
        mask_image = mask_image.reshape(64, 64).unsqueeze(0)
        if self.edw_aggregate_mask is None:
            self.edw_aggregate_mask = mask_image
        else:
            self.edw_aggregate_mask = torch.max(self.edw_aggregate_mask, mask_image)                  
            
        if idx == len(self.edw_token_idx)-1:
            overwatch.info(f"Save Automatic Mask", ctx_level=1)
            # os.makedirs(self.save_dir, exist_ok=True)
            if self.save_dir is not None:
                str_edw_token_idx = "_".join([str(i) for i in self.edw_token_idx])
                save_image(mask_image, os.path.join(self.save_dir, f"auto_mask_{self.image_name}_edw_thr_{self.edw_threshold}_edw_token_id_{str_edw_token_idx}.jpg"))
                
        return image
    
def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x # cross-attention이 아니라면, context는 그냥 x로 설정
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale 

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            
            #NOTE: 단지 stable diffusion의 attention과 다른 점은 이곳!!
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet, # place_in_unet = ["down", "mid", "up"]
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            # if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
            if 'Attention' in net.__class__.__name__:
                # print(net.__class__.__name__)
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    # print(cross_att_count)
    editor.num_att_layers = cross_att_count

# Copyright from Prompt-to-prompt(https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py)
def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
    
# Copyright from PNP-Diffusers(https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp_utils.py)
def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                uc_hidden_states, c_hidden_states = hidden_states.chunk(2)
                source_batch_size = int(uc_hidden_states.shape[0] // 3)
                # inject unconditional
                uc_hidden_states[source_batch_size:2 * source_batch_size] = uc_hidden_states[2 * source_batch_size:]
                # inject conditional
                c_hidden_states[source_batch_size:2 * source_batch_size] = c_hidden_states[2 * source_batch_size:]
                hidden_states = torch.cat([uc_hidden_states, c_hidden_states], dim=0)

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
    
# Copyright from PNP-Diffusers(https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp_utils.py)
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)