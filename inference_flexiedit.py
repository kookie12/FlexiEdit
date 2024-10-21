import os
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.io import read_image
from diffusers import DDIMScheduler, StableDiffusionPipeline
from flexiedit.diffuser_utils import FlexiEditPipeline
from flexiedit.ti_diffuser_utils import TIGuidedPipeline
from flexiedit.attention_register import regiter_attention_editor_diffusers, register_conv_control_efficient, FE_AttentionStore
from flexiedit.attention_utils import FE_MutualSelfAttentionControl, FE_MutualSelfAttentionControlMaskAuto, FE_UnifiedSelfAttentionControl
import fire
from flexiedit.frequency_utils import get_freq_filter, freq_2d
from flexiedit.get_edited_words import find_edited_phrases
import numpy as np
from flexiedit.utils import slerp_tensor, load_512, txt_draw, latent2image, add_text_to_image, draw_mask, tensor2numpy, make_grid
from PIL import Image
import random
from pytorch_lightning import seed_everything
from box import Box
from collections import OrderedDict
import yaml

# Initialize Overwatch =>> Wraps `logging.Logger`
from overwatch import initialize_overwatch
overwatch = initialize_overwatch(__name__)

''' define hyperparameters '''
# low-pass filter settings
filter_type= "gaussian" #"butterworth" 
n= 4 # gaussian parameter
# Sampling process settings
global alpha, reinversion_step, d_s, d_t, refined_step, masa_step_original, masa_step_target_branch, masa_step_retarget_branch
alpha = 0.7
d_t= 0.3
d_s= 0.3
refined_step = 0 
masa_step_original = 4
masa_step_target_branch = 51
masa_step_retarget_branch = 0

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_image(image_path, device):
    """ Load an image, resize and center crop it. """
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = T.Resize(512)(image)
    image = T.CenterCrop(512)(image)
    image = image.to(device)
    return image

def get_word_inds(text: str, word_place: int, tokenizer):
    """ Get indices of words in the provided text using tokenizer. """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out

def freq_exp(feat, mode, user_mask, auto_mask):
    """ Frequency manipulation for latent space. """
    feat = feat.view(4,1,64,64)
    f_shape = feat.shape # 1, 4, 64, 64
    LPF = get_freq_filter(f_shape, feat.device, filter_type, n, d_s, d_t) # d_s, d_t
    f_dtype = feat.dtype
    feat_low, feat_high, feat_sum, feat_low_alpha, feat_high_alpha, feat_low_alpha_high, feat_high_alpha_low, x_alpha_high_alpha_low = freq_2d(feat.to(torch.float64), LPF, alpha)
    feat_low = feat_low.to(f_dtype)
    feat_high = feat_high.to(f_dtype)
    feat_sum = feat_sum.to(f_dtype)
    feat_low_alpha = feat_low_alpha.to(f_dtype)
    feat_high_alpha = feat_high_alpha.to(f_dtype)
    feat_low_alpha_high = feat_low_alpha_high.to(f_dtype)
    feat_high_alpha_low = feat_high_alpha_low.to(f_dtype)

    # latent LPF
    latent_low = feat_low.view(1,4,64,64)
    # latent HPF
    latent_high = feat_high.view(1,4,64,64)
    # latent SUM (original)
    latent_sum = feat_sum.view(1,4,64,64)
    
    # latent_low_alpha = feat_low_alpha.view(1,4,64,64)
    # latent_high_alpha = feat_high_alpha.view(1,4,64,64)
    latent_low_alpha_high = feat_low_alpha_high.view(1,4,64,64)
    latent_high_alpha_low = feat_high_alpha_low.view(1,4,64,64)
    
    mask = torch.zeros_like(latent_sum)
    if mode == "auto_mask":
        auto_mask = auto_mask.unsqueeze(1) # [1,64,64] => [1,1,64,64]
        mask = auto_mask.expand_as(latent_sum) # [1,1,64,64] => [1,4,64,64]
        
    elif mode == "user_mask":
        bbx_start_point, bbx_end_point = user_mask
        mask[:, :, bbx_start_point[1]//8:bbx_end_point[1]//8, bbx_start_point[0]//8:bbx_end_point[0]//8] = 1
        
    latents_shape = latent_sum.shape
    random_gaussian = torch.randn(latents_shape, device=latent_sum.device)
    
    # Apply gaussian scaling
    g_range = random_gaussian.max() - random_gaussian.min()
    l_range = latent_low_alpha_high.max() - latent_low_alpha_high.min()
    random_gaussian = random_gaussian * (l_range/g_range)

    # No scaling applied. If you wish to apply scaling to the mask, replace the following lines accordingly.
    s_range, r_range, s_range2, r_range2 = 1, 1, 1, 1
        
    latent_mask_h = latent_sum * (1 - mask) + (latent_low_alpha_high + (1-alpha)*random_gaussian) * (s_range/r_range) *mask # edit 할 부분에 high frequency가 줄어들고 가우시안 더하기
    latent_mask_l = latent_sum * (1 - mask) + (latent_high_alpha_low + (1-alpha)*random_gaussian) * (s_range2/r_range2) *mask # edit 할 부분에 low frequency가 줄어들고 가우시안 더하기
    
    return latent_mask_h, latent_mask_l, latent_sum # latent_low, latent_high, latent_sum

def setup_editor_and_params(masa_step, masa_layer, inject_uncond, inject_cond, save_path):
    # default setting
    editor = None
    npi = False
    npi_interp = 0
    prox = None
    quantile = None
    guidance_scale = [1, 7.5]
        
    #NOTE: In here, we set masa_step_target_branch to 51, which means the feature injection will not be performed.
    editor = FE_MutualSelfAttentionControl(masa_step_target_branch, masa_layer, inject_uncond=inject_uncond, inject_cond=inject_cond)
    npi = False #
    npi_interp = 0
    prox = None
    quantile = None

    return editor, npi, npi_interp, prox, quantile, guidance_scale

def main(
    start_noise_interp: float = 0.0,
    model_path = "../CompVis/stable-diffusion-v1-5",
    out_dir: str = None,
    source_image_path: str = None,
    source_prompt = None,
    target_prompt = None,
    scale: float = 7.5,
    inv_scale: float = 1,
    query_intermediate: bool = False,
    masa_step: int = 4,
    masa_layer: int = 10,
    inject_uncond: str = "src",
    inject_cond: str = "src",
    prox_step: int = 0,
    prox: str = None,
    quantile: float = 0.7,
    npi: bool = False,
    npi_interp: float = 0,
    npi_step: int = 0,
    num_inference_steps: int = 50,
    editing_type: str = None,
    reinversion_steps: int = 20,
    cuda_device: str = "cuda:0",
    blended_word: str = None, 
    bbx_start_point=None,
    bbx_end_point=None
):  
    
    device = torch.device(cuda_device) if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    
    overwatch.info(f"[Frozen]  🥶 ==>> Loading FlexiEdit from [bold]{model_path}[/] Checkpoint")
    flexiedit = FlexiEditPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)
    
    source_image = load_image(source_image_path, device) # Normalize to the range [-1, 1]
    
    #NOTE: save_path
    save_path = os.path.join(out_dir, source_image_path.split("/")[-1].split(".")[0], target_prompt)
    os.makedirs(save_path, exist_ok=True)
    sample_count = len(os.listdir(save_path))
    save_path = os.path.join(save_path, f"sample_{sample_count+1}")
    os.makedirs(save_path, exist_ok=True)
    prompts = [source_prompt, target_prompt]

    setup_seed() 
    
    res=32
    save_mask_timestep = 10
    threshold = 0.05 # (0.02-0.15)
    output_dir = save_path
    edw_threshold = 0.15
    
    #NOTE: Invert
    ind = get_word_inds(source_prompt, blended_word[0], flexiedit.tokenizer)
    assert len(ind) != 0, "The object name must in the source prompt."
    
    editor = FE_AttentionStore(res=res, ref_token_idx=ind, save_mask_timestep=save_mask_timestep, 
                                threshold=threshold, save_dir=output_dir, image_name="app")
    
    #NOTE: automatic mask 
    if bbx_start_point == None and bbx_end_point == None:
        edited_words = find_edited_phrases(source_image_path, source_prompt, target_prompt)
        edw_indices = []
        
        for edited_word in edited_words:
            if " " in edited_word:
                for word in edited_word.split():
                    ind = get_word_inds(target_prompt, word, flexiedit.tokenizer)
                    edw_indices.append(ind[0])
            else:
                ind = get_word_inds(target_prompt, edited_word, flexiedit.tokenizer)
                edw_indices.append(ind[0])
        assert len(edw_indices) != 0, "error: edited words is not detected"
        editor.edw_token_idx = edw_indices
        editor.save_edw_mask_timestep = 1
        editor.edw_threshold = edw_threshold
        
    regiter_attention_editor_diffusers(flexiedit, editor)
    
    overwatch.info(f"1st stage 🔥 ==>> Inverting source image")
    inv_start_code, inv_latents_list = flexiedit.invert(source_image,
                                            source_prompt,
                                            guidance_scale=inv_scale, # Here, if a tuple (a, b) is provided, proxinpi is used; currently set to 1.
                                            num_inference_steps=num_inference_steps,
                                            reinversion_steps = 0,
                                            return_intermediates=True,
                                            cuda_device=cuda_device,)
    
    overwatch.info(f"[Finish] ==>> Inverting source image", ctx_level=1)
    mask_appearance = editor.get_aggregate_mask()
    flexiedit.set_app_mask(mask_appearance)
        
    #NOTE: Automatic mask 
    if bbx_start_point == None and bbx_end_point == None:
        overwatch.info(f"2nd stage 🔥 ==>> Refine DDIM latent using Automatic mask")
        edw_mask = editor.get_edw_aggregate_mask()  
        latent_mask_h, latent_mask_l, latent_sum = freq_exp(inv_start_code, "auto_mask", None, edw_mask)
    
    #NOTE: User provide user-defined mask
    else:
        edw_mask=None 
        overwatch.info(f"2nd stage 🔥 ==>> Refine DDIM latent using user-defined mask")
        latent_mask_h, latent_mask_l, latent_sum = freq_exp(inv_start_code, "user_mask", (bbx_start_point, bbx_end_point), None)
    
    # visualize latents_list
    new_latents_list = [latent2image(flexiedit.vae, latent) for latent in inv_latents_list]
    
    # If you want to save the latent visualization, uncomment the following code.
    # for i, latent in enumerate(new_latents_list):
    #     save_image(latent, os.path.join(save_path, f"latent_{i}.png"))
    
    # concat latents_list to one image using numpy
    new_latents_list = [latent[0] for latent in new_latents_list][::5]
    concat_latents_list = np.concatenate(new_latents_list, axis=1)
    
    # save decoded latent visualization across all steps
    # Image.fromarray(concat_latents_list).save(os.path.join(save_path, "decoded_latent_visualize.png"))
    # source_image_mask = torch.from_numpy(draw_mask(tensor2numpy(source_image), bbx_start_point, bbx_end_point))
    
    if start_noise_interp > 0:
        random_code = flexiedit.prepare_latents(
            start_code.shape[0],
            start_code.shape[1], 
            512, 512, 
            dtype=start_code.dtype, 
            device=start_code.device, 
            generator=torch.Generator("cuda").manual_seed(42))
        start_code = torch.cat([
            start_code,
            slerp_tensor(start_noise_interp, start_code, random_code)
        ], dim=0)
    else:
        latent_sum = latent_sum.expand(len(prompts), -1, -1, -1)
        latent_mask_h = latent_mask_h.expand(len(prompts), -1, -1, -1)
        latent_mask_l = latent_mask_l.expand(len(prompts), -1, -1, -1)

    if prox == "none" or prox == "None":
        prox = None

    config0 = Box()
    config0.model_path = model_path
    config0.save_path = save_path
    config0.source_image_path = source_image_path
    config0.source_prompt = source_prompt
    config0.target_prompt = target_prompt
    
    config1 = Box()
    config1.adain_start_step = 30
    config1.adain_end_step = 50
    config1.alpha = alpha
    config1.n = n
    config1.reinversion_step = reinversion_steps
    config1.d_s = d_s
    config1.d_t = d_t
    config1.refined_step = refined_step
    user_box = OrderedDict()
    user_box["bbx_start_point"] = str(bbx_start_point)
    user_box["bbx_end_point"] = str(bbx_end_point)
    config1.user_bbx = user_box
    
    overwatch.info(f"3rd stage 🔥 ==>> Generating image from DDIM latent")
    overwatch.info(f"[Generate Image] [bold]FlexiEdit is performing", ctx_level=1)
    editor, npi, npi_interp, prox, quantile, guidance_scale = setup_editor_and_params(masa_step, masa_layer, inject_uncond, inject_cond, save_path)
    regiter_attention_editor_diffusers(flexiedit, editor) 
    output_mid = flexiedit(prompts,
                        latents=latent_mask_h, # start_code
                        num_inference_steps=num_inference_steps,
                        guidance_scale=[1, scale],
                        neg_prompt=source_prompt if npi else None,
                        prox=prox,
                        prox_step=prox_step,
                        quantile=quantile,
                        npi_interp=npi_interp,
                        npi_step=npi_step, 
                        ref_intermediate_latents=None, 
                        mode="FlexiEdit", #mode,
                        latent_filter=[latent_sum, None, None, None],
                        params={"n": n,
                                    "alpha": alpha,
                                    "reinversion_step": reinversion_steps,
                                    "d_s": d_s,
                                    "d_t": d_t,
                                    "refined_step": refined_step,
                                    "user_mask": [bbx_start_point, bbx_end_point],
                                    "auto_mask": edw_mask,
                                    "callback": flexiedit.get_adain_app_callback(config1.adain_start_step, config1.adain_end_step),
                                    "cuda_device": cuda_device,
                                    }
    )
    
    source_image_2 = ((output_mid[1] - 0.5)*2).unsqueeze(0)
    
    ind = get_word_inds(target_prompt, blended_word[1], flexiedit.tokenizer)
    assert len(ind) != 0, "The object name must in the target prompt."
    
    editor = FE_AttentionStore(res=res, ref_token_idx=ind, save_mask_timestep=save_mask_timestep, 
                                threshold=threshold, save_dir=output_dir, image_name="struct")
    regiter_attention_editor_diffusers(flexiedit, editor) 
    
    #NOTE: Re-Inversion
    overwatch.info(f"4th Stage 🔥 ==>> Starting re-inversion process using FlexiEdit only")
    reinv_start_code, reinv_latents_list = flexiedit.invert(source_image_2,
                                            source_prompt,
                                            guidance_scale=inv_scale,
                                            num_inference_steps=num_inference_steps, # reinversion step
                                            reinversion_steps = reinversion_steps,
                                            return_intermediates=True,
                                            cuda_device=cuda_device,
                                            mode='REINVERSION')
    
    overwatch.info(f"[Finish] ==>> Re-Inverting source image", ctx_level=1)
    mask_struct = editor.get_aggregate_mask()
    flexiedit.set_struct_mask(mask_struct)
    
    mask_save_dir = os.path.join(save_path, "REINVERSION")
    reinv_flexiedit_editor = FE_MutualSelfAttentionControl(masa_step_retarget_branch, masa_layer, inject_uncond=inject_uncond, inject_cond=inject_cond)
    regiter_attention_editor_diffusers(flexiedit, reinv_flexiedit_editor)
    
    npi_2nd=False
    npi_interp_2nd=0
    
    #NOTE: FlexiEdit Re-Target Branch
    overwatch.info(f"5th Stage 🔥 ==>> Starting re-sampling process using FlexiEdit only")
    output_reinversion = flexiedit(prompts,
                        latents=reinv_start_code.expand(len(prompts), -1, -1, -1), # start_code=[2, 4, 64, 64]
                        num_inference_steps=reinversion_steps,
                        guidance_scale=[1, scale],
                        neg_prompt=source_prompt if npi_2nd else None,
                        prox=prox,
                        prox_step=prox_step,
                        ref_intermediate_latents=None, # latents_list
                        quantile=quantile,
                        npi_interp=npi_interp_2nd,
                        npi_step=npi_step,
                        mode='REINVERSION',
                        latent_filter=[inv_latents_list[reinversion_steps].expand(len(prompts), -1, -1, -1), None, None, None],
                        params={"n": n,
                                "alpha": alpha,
                                "reinversion_step": reinversion_steps,
                                "d_s": d_s,
                                "d_t": d_t,
                                "refined_step": refined_step,
                                "user_mask": [bbx_start_point, bbx_end_point],
                                "auto_mask": edw_mask,
                                "callback": flexiedit.get_adain_bg_callback(config1.adain_start_step, config1.adain_end_step),
                                "cuda_device": cuda_device,
                                })

    #NOTE: For advanced background fidelity, we utilize unified self-attention control in FlexiEdit
    config2 = Box()
    config2.latent_blend_type = "bg"
    config2.latent_blend_step = 0 
    config2.adain_start_step = reinversion_steps//2 
    config2.adain_end_step = reinversion_steps 
    config2.conv_injection_t = 40
    config2.app_start_step = 4 
    config2.app_end_step = reinversion_steps 
    config2.app_start_layer = 10
    config2.struct_start_step = 0 
    config2.struct_end_step = 25 
    config2.struct_start_layer = 0
    config2.contrast_strength = 1.67 # not used
    config2.injection_step = 1
    config2.appearance_invert_flag = True
    config2.struct_invert_flag = True
    config2.mode = 'both'
    
    # set prompt
    # prompts = [appearance_prompt, target_prompt, struct_prompt]
    negative_prompt = "ugly, blurry, black, low res, unrealistic"
    prompt = [source_prompt, target_prompt, target_prompt]
    appearance_neg_prompt = source_prompt if config2.appearance_invert_flag else negative_prompt
    struct_neg_prompt = target_prompt if config2.struct_invert_flag else negative_prompt
    neg_prompts = [appearance_neg_prompt, struct_neg_prompt, struct_neg_prompt]
    
    # set scale
    text_scale, guidance_scale = 7.5, 7.5
    app_scale = inv_scale if config2.appearance_invert_flag else text_scale
    struct_scale = inv_scale if config2.struct_invert_flag else text_scale
    scale = [app_scale, guidance_scale, struct_scale]
    
    # latent concat
    # new_start_code = torch.cat([inv_start_code, reinv_start_code, reinv_start_code], dim=0) 
    new_start_code = torch.cat([inv_latents_list[reinversion_steps], reinv_start_code, reinv_start_code], dim=0) 
    # new_start_code = torch.cat([inv_latents_list[len(inv_latents_list) - reinversion_steps], reinv_start_code, reinv_start_code], dim=0) 
    
    editor = FE_UnifiedSelfAttentionControl(appearance_start_step=config2.app_start_step, 
                                        appearance_end_step=config2.app_end_step, 
                                        appearance_start_layer=config2.app_start_layer, 
                                        struct_start_step=config2.struct_start_step, 
                                        struct_end_step=config2.struct_end_step, 
                                        struct_start_layer=config2.struct_start_layer, 
                                        mix_type=config2.mode, 
                                        contrast_strength=config2.contrast_strength, 
                                        injection_step=config2.injection_step)
    regiter_attention_editor_diffusers(flexiedit, editor) 
    
    # hijack the resblock module 
    # injection_step = injection_step if mode == "app" else 1
    conv_injection_timesteps = scheduler.timesteps[:config2.conv_injection_t:config2.injection_step] if config2.conv_injection_t >= 0 else []
    register_conv_control_efficient(flexiedit, conv_injection_timesteps)
    
    #NOTE: New version of FlexiEdit: Re-Target Branch
    overwatch.info(f"6th Stage 🔥 ==>> New version of FlexiEdit!!")
    image_results = flexiedit(prompts,
                        latents=new_start_code,
                        num_inference_steps=reinversion_steps,
                        mode="new_FlexiEdit",
                        new_params={
                            "scale": scale,
                            "neg_prompts": neg_prompts,
                            "ref_intermediate_latents_app": inv_latents_list[:reinversion_steps+1] if config2.appearance_invert_flag else None,
                            "ref_intermediate_latents_struct": reinv_latents_list if config2.struct_invert_flag else None,
                            "callback": flexiedit.get_adain_bg_callback(config2.adain_start_step, config2.adain_end_step),
                            "latent_blend_type": config2.latent_blend_type,
                            "latent_blend_step": config2.latent_blend_step,
                            "cuda_device": cuda_device
                        }
    )

    out_image_mid = tensor2numpy(torch.cat([source_image * 0.5 + 0.5, output_mid], dim=0))
    out_image_reinversion = tensor2numpy(torch.cat([source_image * 0.5 + 0.5, output_reinversion], dim=0))
    
    # swap ordering
    grid1 = image_results[0].unsqueeze(dim=0)
    grid2 = image_results[1].unsqueeze(dim=0)
    grid3 = image_results[2].unsqueeze(dim=0)
    
    new_image_results = torch.cat([grid3, grid2], dim=0)
    out_image_new = tensor2numpy(torch.cat([source_image * 0.5 + 0.5, new_image_results], dim=0))
    
    
    #NOTE: automatic mask 
    if bbx_start_point != None and bbx_end_point != None:
        out_image_mask = draw_mask(out_image_mid, bbx_start_point, bbx_end_point)
        
    else:
        # draw mask in out_image_mask
        grid = make_grid(edw_mask)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
        _, c, h, w = source_image.shape
        
        # upscale ndarr (64,64,3) to the size of source_image (512,512,3)
        upscale_ndarr = Image.fromarray(ndarr.cpu().numpy().astype(np.uint8)).resize((w, h))
        red_mask = Image.new("RGBA", (w, h), (255, 0, 0, 100))
        origin_image = Image.fromarray(out_image_mid[:, 0:h, :3].astype(np.uint8))
        
        
        masked_image = Image.composite(origin_image, upscale_ndarr, red_mask)
        out_image_mid[:, 0:h, :3] = np.array(masked_image)
        out_image_mask = out_image_mid
    
    # for latent
    latent_512 = latent2image(flexiedit.vae, inv_start_code)[0]
    latent_mid_512 = latent2image(flexiedit.vae, latent_mask_h)[0]
    latent_reinvserion_512 = latent2image(flexiedit.vae, reinv_start_code)[0]
    
    image_instruct_00 = txt_draw(f"Model", v="center", h="center", target_size=[512, 100])
    image_instruct_01 = txt_draw(f"DDIM Latent", v="center", h="center", target_size=[512, 100])
    image_instruct_02 = txt_draw(f"Original Image \n (First row: +edited mask)", v="center", h="center", target_size=[512, 100])
    image_instruct_03 = txt_draw(f"Reconstruction Image", v="center", h="center", target_size=[512, 100])
    image_instruct_04 = txt_draw(f"Edited Image", v="center", h="center", target_size=[512, 100])
    
    if bbx_start_point == None and bbx_end_point == None:
        image_instruct_1 = txt_draw(f"FlexiEdit: Target branch\n (auto mask version!!) \n\n Edited image => I_mid \n\n "
                                    f"source_prompt: {source_prompt} \n target_prompt: {target_prompt} \n "
                                    f"edited_words: {str(edited_words)} \n edw_threshold: {edw_threshold} \n"
                                    f"alpha: {alpha} \n bbx_start/bbx_end_point: {bbx_start_point}, {bbx_end_point} \n "
                                    f"adain_start/end_step: {config1.adain_start_step}, {config1.adain_end_step}" ) 
    else:
        image_instruct_1 = txt_draw(f"FlexiEdit: Target branch\n (user-defied mask version!!) \n\n Edited image => I_mid \n\n "
                                    f"source_prompt: {source_prompt} \n target_prompt: {target_prompt} \n "
                                    f"alpha: {alpha} \n bbx_start/bbx_end_point: {bbx_start_point}, {bbx_end_point} \n "
                                    f"adain_start/end_step: {config1.adain_start_step}, {config1.adain_end_step} \n ") 
        
    image_instruct_2 = txt_draw(f"FlexiEdit: Retarget branch\n (after Re-inversion)\n\n Edited image => I_tar \n\nreinversion timestep: {reinversion_steps} \nmasastep_original: {masa_step_original} \nmasastep_target_branch: {masa_step_target_branch} \nmasastep_retarget_branch: {masa_step_retarget_branch} \nrefined step: {refined_step}")
    image_instruct_3 = txt_draw(f"FlexiEdit: Retarget branch\n(Advanced version!!) \n\n Edited image => I_tar_2 \n\nlatent_blend_step: {config2.latent_blend_step} \nadain_start_step: {config2.adain_start_step}, adain_end_step: {config2.adain_end_step} \napp_start_step: {config2.app_start_step}, app_end_step: {config2.app_end_step} \napp_struct_start_step: {config2.struct_start_step}, app_struct_end_step: {config2.struct_end_step}")    
    
    image_instruct_null = txt_draw(f"")
    
    # out_image_instruct = np.concatenate([image_instruct_1, image_instruct_2, image_instruct_3], axis=1)
    low_0 = np.concatenate([image_instruct_00, image_instruct_01, image_instruct_02, image_instruct_03, image_instruct_04], axis=1)
    low_1 = np.concatenate([image_instruct_1, latent_mid_512, out_image_mask], axis=1)
    low_2 = np.concatenate([image_instruct_2, latent_reinvserion_512, out_image_reinversion], axis=1)
    low_3 = np.concatenate([image_instruct_3, latent_reinvserion_512, out_image_new], axis=1)
    
    total = np.concatenate([low_0, low_1, low_2, low_3], axis=0)
    
    new_sample_count = sample_count + 1
    filename = f'{save_path}/{new_sample_count}_reinv_{reinversion_steps}_points_{str(bbx_start_point)}_{str(bbx_end_point)}.jpg' # _ref_{ref_token_idx}_cur_{cur_token_idx}
    Image.fromarray(total).save(filename)
    
    config = Box()
    config._global_setting = config0
    config.flexiedit = config1
    config.flexiedit_advanced = config2
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/config.yaml", "w") as file:
        yaml.dump(config.to_dict(), file, default_flow_style=False)
    
    print("Syntheiszed images are saved in", os.path.join(out_dir, filename))
    print("Real image | Reconstructed image | Edited image")

if __name__ == "__main__":
    fire.Fire(main)
