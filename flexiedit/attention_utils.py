import os
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .attention_register import AttentionBase
from torchvision.utils import save_image
from overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Copyright from MasaCtrl(https://github.com/TencentARC/MasaCtrl/blob/main/masactrl/masactrl_utils.py)
class FE_MutualSelfAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, inject_uncond="src", inject_cond="src"):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16)) # (4, 16) decoder layer에서만 적용
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps)) # 50~4 timestep에서만 적용
        self.inject_uncond = inject_uncond
        self.inject_cond = inject_cond
        overwatch.info(f"FlexiEdit at denoising steps: {self.step_idx}", ctx_level=2)
        overwatch.info(f"FlexiEdit at U-Net layers:  {self.layer_idx}", ctx_level=2)
        # print("step_idx: ", self.step_idx)
        # print("layer_idx: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale") # 
        # sim = sim/0.7
        attn = sim.softmax(-1)
        
        # gumbel softmax
        # attn = F.gumbel_softmax(sim, tau=0.5, hard=True, dim=-1)
        
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2) # qu=[16,1024,80], q=[32,1024,80]
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2) # attnu=[16,1024,1024], attn=[32,1024,1024]

        # if self.inject_uncond == "src":
        #     out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_uncond == "joint":
        #     out_u = self.attn_batch(qu, ku, vu, None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_uncond == "none":  # no swap
        #     out_u = torch.cat([
        #         self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs),
        #         self.attn_batch(qu[num_heads:], ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)], dim=0)
        # elif self.inject_uncond == "tar":  # this should never be used
        #     out_u = self.attn_batch(qu, ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # else:
        #     raise NotImplementedError
        # if self.inject_cond == "src":
        #     out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_cond == "joint":
        #     out_c = self.attn_batch(qc, kc, vc, None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # elif self.inject_cond == "none":  # no swap
        #     out_c = torch.cat([
        #         self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs),
        #         self.attn_batch(qc[num_heads:], kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)], dim=0)
        # elif self.inject_cond == "tar":  # this should never be used
        #     out_c = self.attn_batch(qc, kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # else:
        #     raise NotImplementedError
        # out = torch.cat([out_u, out_c], dim=0)

        # out_u_0, out_c_0=[1,1024,640]
        out_u_0 = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_0 = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        if self.inject_uncond == "src":
            out_u_1 = self.attn_batch(qu[num_heads:], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_uncond == "joint":
            out_u_1 = self.attn_batch(qu[num_heads:], ku, vu, None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_uncond == "none" or self.inject_uncond == "tar":  # no swap
            out_u_1 = self.attn_batch(qu[num_heads:], ku[num_heads:], vu[num_heads:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            raise NotImplementedError
        if self.inject_cond == "src":
            out_c_1 = self.attn_batch(qc[num_heads:], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_cond == "joint":
            out_c_1 = self.attn_batch(qc[num_heads:], kc, vc, None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.inject_cond == "none" or self.inject_cond == "tar":  # no swap
            out_c_1 = self.attn_batch(qc[num_heads:], kc[num_heads:], vc[num_heads:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            raise NotImplementedError
        out = torch.cat([out_u_0, out_u_1, out_c_0, out_c_1], dim=0) # [4, 1024, 640]

        return out

# Copyright from Ti-Guided-Edit(https://github.com/Kihensarn/TI-Guided-Edit/blob/main/utils/masactrl.py)
class FE_UnifiedSelfAttentionControl(AttentionBase):
    def __init__(self, appearance_start_step=10, appearance_end_step=10, appearance_start_layer=10, 
                 struct_start_step=30, struct_end_step=30, struct_start_layer=8, mix_type="both", 
                 contrast_strength=1.67, injection_step=1):
        super().__init__()
        self.mix_type = mix_type
        self.contrast_strength = contrast_strength
        self.injection_step = injection_step
        self.appearance_start_step = appearance_start_step
        self.appearance_end_step = appearance_end_step
        self.appearance_start_layer = appearance_start_layer
        self.appearance_layer_idx = list(range(appearance_start_layer, 16))
        self.appearance_step_idx = list(range(appearance_start_step, appearance_end_step))

        self.struct_end_step = struct_end_step
        self.struct_start_step = struct_start_step
        self.struct_start_layer = struct_start_layer
        self.struct_layer_idx = list(range(struct_start_layer, 16))
        self.struct_step_idx = list(range(struct_start_step, struct_end_step))

        print("appearance step_idx: ", self.appearance_step_idx)
        print("appearance layer_idx: ", self.appearance_layer_idx)
        print("struct step_idx: ", self.struct_step_idx)
        print("struct layer_idx: ", self.struct_layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def contrast_attn(self, attn_map, contrast_factor):
        attn_mean = torch.mean(attn_map, dim=(0), keepdim=True)
        attn_map = (attn_map - attn_mean) * contrast_factor + attn_mean
        attn_map = torch.clip(attn_map, min=0.0, max=1.0)
        return attn_map

    def attn_batch_app(self, qc, kc, vc, ks, vs, sim, attn, num_heads, contrast_factor, is_rearrange, is_contrast, **kwargs):
        b = qc.shape[0] // num_heads
        qc = rearrange(qc, "(b h) n d -> h (b n) d", h=num_heads)
        kc = rearrange(kc, "(b h) n d -> h (b n) d", h=num_heads)
        vc = rearrange(vc, "(b h) n d -> h (b n) d", h=num_heads)
        ks = rearrange(ks, "(b h) n d -> h (b n) d", h=num_heads)
        vs = rearrange(vs, "(b h) n d -> h (b n) d", h=num_heads)
        sim_source = torch.einsum("h i d, h j d -> h i j", qc, kc) * kwargs.get("scale")
        sim_target = torch.einsum("h i d, h j d -> h i j", qc, ks) * kwargs.get("scale")

        if is_rearrange:
            v = torch.cat([vs, vc], dim=-2)
            C = torch.log2(torch.exp(sim_source).sum(dim=-1) / torch.exp(sim_target).sum(dim=-1))
            sim = torch.cat([sim_target+C.unsqueeze(-1), sim_source], dim=-1)
            attn = sim.softmax(-1)
        else:
            v = vs
            attn = sim_target.softmax(-1)

        if is_contrast:
            attn = self.contrast_attn(attn, contrast_factor)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def unified_attn_batch(self, qc, kc, vc, qs, ks, vs, qa, ka, va, sim, attn, contrast_factor, num_heads, **kwargs):
        b = qc.shape[0] // num_heads
        qc = rearrange(qc, "(b h) n d -> h (b n) d", h=num_heads)
        kc = rearrange(kc, "(b h) n d -> h (b n) d", h=num_heads)
        vc = rearrange(vc, "(b h) n d -> h (b n) d", h=num_heads)
        qs = rearrange(qs, "(b h) n d -> h (b n) d", h=num_heads)
        ks = rearrange(ks, "(b h) n d -> h (b n) d", h=num_heads)
        vs = rearrange(vs, "(b h) n d -> h (b n) d", h=num_heads)
        qa = rearrange(qa, "(b h) n d -> h (b n) d", h=num_heads)
        ka = rearrange(ka, "(b h) n d -> h (b n) d", h=num_heads)
        va = rearrange(va, "(b h) n d -> h (b n) d", h=num_heads)
        v = torch.cat([va, vc], dim=-2)

        sim_source = torch.einsum("h i d, h j d -> h i j", qc, kc) * kwargs.get("scale")
        sim_target_struct = torch.einsum("h i d, h j d -> h i j", qs, ks) * kwargs.get("scale")
        sim_target_app = torch.einsum("h i d, h j d -> h i j", qc, ka) * kwargs.get("scale")

        attn_target_struct = sim_target_struct.softmax(-1) 
        attn_target_struct = self.contrast_attn(attn_target_struct, contrast_factor)
        sim_target = torch.matmul(attn_target_struct, sim_target_app)

        C = torch.log2(torch.exp(sim_source).sum(dim=-1) / torch.exp(sim_target).sum(dim=-1))
        sim = torch.cat([sim_target+C.unsqueeze(-1), sim_source], dim=-1)
        attn = sim.softmax(-1)

        attn = self.contrast_attn(attn, contrast_factor)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # appearance
        out_u_0 = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_0 = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # struct
        out_u_2 = self.attn_batch(qu[num_heads*2:], ku[num_heads*2:], vu[num_heads*2:], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_2 = self.attn_batch(qc[num_heads*2:], kc[num_heads*2:], vc[num_heads*2:], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # target
        out_u_1 = self.attn_batch(qu[num_heads:num_heads*2], ku[num_heads:num_heads*2], vu[num_heads:num_heads*2], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_1 = self.attn_batch(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        
        # determine the mix type of each layer in each step
        if self.cur_step % self.injection_step == 0 and (self.cur_step in self.appearance_step_idx and self.cur_att_layer // 2 in self.appearance_layer_idx) and (self.cur_step in self.struct_step_idx and self.cur_att_layer // 2 in self.struct_layer_idx):
            cur_mix_type = "both"
        elif self.mix_type != "struct" and (self.cur_step in self.appearance_step_idx and self.cur_att_layer // 2 in self.appearance_layer_idx):
            cur_mix_type = "app"
        elif self.mix_type != "app" and (self.cur_step in self.struct_step_idx and self.cur_att_layer // 2 in self.struct_layer_idx):
            cur_mix_type = "struct"
        else:
            cur_mix_type = None

        # attention injection
        if cur_mix_type == "both":
            out_c_1 = self.unified_attn_batch(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], qc[num_heads*2:], kc[num_heads*2:], vc[num_heads*2:], qc[:num_heads], kc[:num_heads], vc[:num_heads], None, attnc, self.contrast_strength, num_heads, **kwargs)
        elif cur_mix_type == "app":
            out_c_1 = self.attn_batch_app(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], kc[:num_heads], vc[:num_heads], None, attnc, num_heads, self.contrast_strength, self.mix_type == "both", self.mix_type == "both", **kwargs)
        elif cur_mix_type == "struct":
            out_c_1 = self.attn_batch(qc[num_heads*2:], kc[num_heads*2:], vc[num_heads:num_heads*2], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            pass
                
        out = torch.cat([out_u_0, out_u_1, out_u_2, out_c_0, out_c_1, out_c_2], dim=0)
        return out


#NOTE: 이건 mask가 주어진 경우에 사용 가능
class FE_MutualSelfAttentionControlMaskAuto(FE_MutualSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, mask_s=None, mask_t=None, mask_save_dir=None):
        """
        Maske-guided MasaCtrl to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps)
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.mask_t = mask_t  # target mask with same shape as source mask
        print("Using mask-guided MasaCtrl")
        if mask_save_dir is not None:
            os.makedirs(mask_save_dir, exist_ok=True)
            save_image(self.mask_s.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_s.png"))
            save_image(self.mask_t.unsqueeze(0).unsqueeze(0), os.path.join(mask_save_dir, "mask_t.png"))

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)
        out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, is_mask_attn=True, **kwargs)

        if self.mask_s is not None and self.mask_t is not None:
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)
            out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out

# NOTE: This will automatically generate a mask if no mask is provided => may result in distortion of the original image
class FE_MutualSelfAttentionControlMaskAuto(FE_MutualSelfAttentionControl):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, thres=0.1, ref_token_idx=[1], cur_token_idx=[1], mask_save_dir=None):
        """
        MasaCtrl with mask auto generation from cross-attention map
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            thres: the thereshold for mask thresholding
            ref_token_idx: the token index list for cross-attention map aggregation
            cur_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps)
        print("using MutualSelfAttentionControlMaskAuto")
        self.thres = thres
        self.ref_token_idx = ref_token_idx
        self.cur_token_idx = cur_token_idx

        self.self_attns = []
        self.cross_attns = []

        self.cross_attns_mask = None
        self.self_attns_mask = None

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)

    def after_step(self):
        self.self_attns = []
        self.cross_attns = []

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if self.self_attns_mask is not None:
            # binarize the mask
            mask = self.self_attns_mask
            thres = self.thres
            mask[mask >= thres] = 1
            mask[mask < thres] = 0
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg])

        attn = sim.softmax(-1)

        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out

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
        return image

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1))

        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        B = q.shape[0] // num_heads // 2
        H = W = int(np.sqrt(q.shape[1]))
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        out_u_source = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_source = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if len(self.cross_attns) == 0:
            self.self_attns_mask = None
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            mask = self.aggregate_cross_attn_map(idx=self.ref_token_idx)  # (2, H, W)
            mask_source = mask[-2]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            self.self_attns_mask = F.interpolate(mask_source.unsqueeze(0).unsqueeze(0), (res, res)).flatten()
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(self.self_attns_mask.shape[0]))
                mask_image = self.self_attns_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_s_{self.cur_step}_{self.cur_att_layer}.png"))
            out_u_target = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_target = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)

        if self.self_attns_mask is not None:
            mask = self.aggregate_cross_attn_map(idx=self.cur_token_idx)  # (2, H, W)
            mask_target = mask[-1]  # (H, W)
            res = int(np.sqrt(q.shape[1]))
            spatial_mask = F.interpolate(mask_target.unsqueeze(0).unsqueeze(0), (res, res)).reshape(-1, 1)
            if self.mask_save_dir is not None:
                H = W = int(np.sqrt(spatial_mask.shape[0]))
                mask_image = spatial_mask.reshape(H, W).unsqueeze(0)
                save_image(mask_image, os.path.join(self.mask_save_dir, f"mask_t_{self.cur_step}_{self.cur_att_layer}.png"))
            # binarize the mask
            thres = self.thres
            spatial_mask[spatial_mask >= thres] = 1
            spatial_mask[spatial_mask < thres] = 0
            out_u_target_fg, out_u_target_bg = out_u_target.chunk(2)
            out_c_target_fg, out_c_target_bg = out_c_target.chunk(2)

            out_u_target = out_u_target_fg * spatial_mask + out_u_target_bg * (1 - spatial_mask)
            out_c_target = out_c_target_fg * spatial_mask + out_c_target_bg * (1 - spatial_mask)

            # set self self-attention mask to None
            self.self_attns_mask = None

        out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
