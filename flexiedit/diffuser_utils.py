"""
Util functions based on Diffuser framework.
"""
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
# from torchvision.utils import save_image
# from torchvision.io import read_image
from diffusers import StableDiffusionPipeline
from flexiedit.utils import slerp_tensor
from flexiedit.frequency_utils import get_freq_filter, freq_1d, freq_2d, freq_3d
from flexiedit.adain import masked_adain
from flexiedit.attention_register import register_time   
from typing import Callable, List
from overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class FlexiEditPipeline(StableDiffusionPipeline):

    def freq_exp(self, feat, mask_mode, user_mask, auto_mask):
        filter_type= "gaussian" 
        feat = feat.chunk(2)[1].view(4,1,64,64)
        f_shape = feat.shape # 1, 4, 64, 64
        LPF = get_freq_filter(f_shape, feat.device, filter_type, self.n, self.d_s, self.d_t)
        f_dtype = feat.dtype
        feat_low, feat_high, feat_sum, feat_low_alpha, feat_high_alpha, feat_low_alpha_high, feat_high_alpha_low, feat_alpha_high_alpha_low = freq_2d(feat.to(torch.float64), LPF, alpha=self.alpha)
        feat_low = feat_low.to(f_dtype)
        feat_high = feat_high.to(f_dtype)
        feat_sum = feat_sum.to(f_dtype)
        feat_low_alpha = feat_low_alpha.to(f_dtype)
        feat_high_alpha = feat_high_alpha.to(f_dtype)
        feat_low_alpha_high = feat_low_alpha_high.to(f_dtype)
        feat_high_alpha_low = feat_high_alpha_low.to(f_dtype)

        # latent LPF
        latent_lpf = feat_low.view(1,4,64,64)
        # latent HPF
        latent_hpf = feat_high.view(1,4,64,64)
        latent_sum = feat_sum.view(1,4,64,64)
        latent_lpf_alpha = feat_low_alpha.view(1,4,64,64)
        latent_hpf_alpha = feat_high_alpha.view(1,4,64,64)
        
        latent_low_alpha_high = feat_low_alpha_high.view(1,4,64,64)
        latent_high_alpha_low = feat_high_alpha_low.view(1,4,64,64)
        
        mask = torch.zeros_like(latent_sum)
        
        if mask_mode == "auto_mask":
            auto_mask = auto_mask.unsqueeze(1) # [1,64,64] => [1,1,64,64]
            mask = auto_mask.expand_as(latent_sum) # [1,1,64,64] => [1,4,64,64]
        
        elif mask_mode == "user_mask":
            start_point, end_point = user_mask
            mask[:, :, start_point[1]//8:end_point[1]//8, start_point[0]//8:end_point[0]//8] = 1
            
        latents_shape = latent_sum.shape
        random_gaussian = torch.randn(latents_shape, device=latent_sum.device)
        
        # Apply gaussian scaling 
        g_range = random_gaussian.max() - random_gaussian.min()
        l_range = latent_low_alpha_high.max() - latent_low_alpha_high.min()
        random_gaussian = random_gaussian * (l_range/g_range)
        
        # No scaling applied. If you wish to apply scaling to the mask, replace the following lines accordingly.
        s_range, r_range, s_range2, r_range2 = 1, 1, 1, 1
        
        latent_mask_h = latent_sum * (1 - mask) + (latent_low_alpha_high + (1-self.alpha)*random_gaussian) * (s_range/r_range) *mask # edit 할 부분에 high frequency가 줄어들고 가우시안 더하기
        latent_mask_l = latent_sum * (1 - mask) + (latent_high_alpha_low + latent_hpf + (1-self.alpha)*random_gaussian) * (s_range2/r_range2) *mask # edit 할 부분에 low frequency가 줄어들고 가우시안 더하기
        
        latent_low_alpha_high = latent_low_alpha_high * (s_range/r_range)
        latent_high_alpha_low = latent_high_alpha_low * (s_range2/r_range2)
        
        return latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low

    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device(self.cuda_device) if torch.cuda.is_available() else torch.device("cpu") # NOTE: cuda:#
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    def latent2image_grad(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)['sample']

        return image  # range [-1, 1]

    def set_app_mask(self, mask):
        self.image_app_mask_64 = mask

    def set_struct_mask(self, mask):
        self.image_struct_mask_64 = mask
        
    def set_edit_mask(self, mask):
        self.image_edit_mask_64 = mask

    def get_adain_app_callback(self, start_step, end_step):
        self.adain_start_step = start_step
        self.adain_end_step = end_step
        # print("adain step range: ", [self.adain_start_step, self.adain_end_step])
        #overwatch.info(f"[Finish] ==>> Inverting source image", ctx_level=1)
        overwatch.info(f"Applying AdaIN using the computed appearance masks ==>> adain step range: [{self.adain_start_step}, {self.adain_end_step}]", ctx_level=1)

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            # Apply AdaIN operation using the computed masks
            if self.adain_start_step <= st < self.adain_end_step:
                # latents[1] = masked_adain(latents[0], latents[1], self.image_app_mask_64, self.image_app_mask_64)
                latents[1] = masked_adain(latents[0], latents[1], 1 - self.image_app_mask_64, 1 - self.image_app_mask_64)
                
        return callback

    def get_adain_bg_callback(self, start_step, end_step):
        self.adain_start_step = start_step
        self.adain_end_step = end_step
        # print("adain step range: ", [self.adain_start_step, self.adain_end_step])
        overwatch.info(f"Applying AdaIN using the computed background masks ==>> adain step range: [{self.adain_start_step}, {self.adain_end_step}]", ctx_level=1)

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            # Apply AdaIN operation using the computed masks
            if self.adain_start_step <= st < self.adain_end_step:
                # latents[1] = masked_adain(latents[0], latents[1], self.image_app_mask_64, self.image_struct_mask_64)
                latents[1] = masked_adain(latents[0], latents[1], 1 - self.image_app_mask_64, 1 - self.image_struct_mask_64)
                
        return callback

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        prox=None,
        prox_step=0,
        quantile=0.7,
        npi_interp=0,
        npi_step=0,
        mode=None,
        latent_filter=None,
        params=None,
        new_params=None,
        **kwds):
                                        
        if params is not None:
            self.n = params["n"]
            self.alpha = params["alpha"]
            self.reinversion_step = params["reinversion_step"]
            self.d_s = params["d_s"]
            self.d_t = params["d_t"]
            self.refined_step = params["refined_step"]
            self.user_mask = params["user_mask"] # [start_point, end_point]
            self.auto_mask = params["auto_mask"] # edw_mask
            self.callback = params["callback"]
            self.cuda_device = params["cuda_device"]
        
        elif new_params is not None:
            self.scale = new_params["scale"]
            self.neg_prompts = new_params["neg_prompts"]
            self.ref_intermediate_latents_app = new_params["ref_intermediate_latents_app"]    
            self.ref_intermediate_latents_struct = new_params["ref_intermediate_latents_struct"]
            self.callback = new_params["callback"]
            self.latent_blend_type = new_params["latent_blend_type"]
            self.latent_blend_step = new_params["latent_blend_step"]
            self.cuda_device = new_params["cuda_device"]
            
        DEVICE = torch.device(self.cuda_device) if torch.cuda.is_available() else torch.device("cpu") # NOTE: cuda:#
        if isinstance(prompt, list) and mode is not 'new_FlexiEdit':
            batch_size = len(prompt)
        elif isinstance(prompt, str) and mode is not 'new_FlexiEdit':
            if batch_size > 1:
                prompt = [prompt] * batch_size
                
        elif isinstance(prompt, list) and mode == 'new_FlexiEdit':
            batch_size = 3
            prompt = [prompt] * batch_size
        
        if isinstance(guidance_scale, (tuple, list)):
            assert len(guidance_scale) == 2
            # guidance_scale_batch = torch.tensor(guidance_scale, device=DEVICE).reshape(2, 1, 1, 1)
            guidance_scale_0, guidance_scale_1 = guidance_scale[0], guidance_scale[1]
            guidance_scale = guidance_scale[1]
            do_separate_cfg = True
        else:
            # guidance_scale_batch = torch.tensor([guidance_scale], device=DEVICE).reshape(1, 1, 1, 1)
            do_separate_cfg = False

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        # overwatch.info(f"text embeddings shape after text_encoder: [bold]{text_embeddings.shape}", ctx_level=1)

        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            # uc_text = "ugly, tiling, poorly drawn hands, poorly drawn feet, body out of frame, cut off, low contrast, underexposed, distorted face"
            if npi_interp > 0:
                assert neg_prompt is not None, "Please provide negative prompt for NPI."
                null_embedding = self.tokenizer(
                    [""] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                neg_embedding = self.tokenizer(
                    [neg_prompt] * 1,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                neg_embedding = self.text_encoder(neg_embedding.input_ids.to(DEVICE))[0]
                # unconditional_embeddings = (1-npi_interp) * npi_embedding + npi_interp * null_embedding
                unconditional_embeddings = slerp_tensor(npi_interp, neg_embedding, null_embedding)
                # unconditional_embeddings = unconditional_embeddings.repeat(batch_size, 1, 1)
                unconditional_embeddings = torch.cat([neg_embedding, unconditional_embeddings], dim=0)
            else:
                unconditional_input = self.tokenizer(
                    [uc_text] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            # text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            if npi_step > 0:
                null_embedding = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=77,
                    return_tensors="pt"
                )
                null_embedding = self.text_encoder(null_embedding.input_ids.to(DEVICE))[0]
                text_embeddings_null = torch.cat([null_embedding, text_embeddings], dim=0)
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
            else:
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)
                
        # overwatch.info(f"latent shape: [bold]{latents.shape}", ctx_level=1)
        # iterative sampling
        # self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_timesteps(50) 
        
        latents_list = [latents]
        pred_x0_list = [latents]
        
        if mode == "REINVERSION" or mode == "new_FlexiEdit":
            self.scheduler.timesteps = self.scheduler.timesteps[50 - num_inference_steps:]
         
        if mode == "ProxiMasaCtrl" or mode == "FlexiEdit" or mode == "REINVERSION" or mode == "hpf":   
            # added by kookie 24.02.20
            latent_sum, latent_lpf, latent_hpf, latent_mask = latent_filter
            
            #NOTE: DDIM Sampler
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                if ref_intermediate_latents is not None:
                    # note that the batch_size >= 2
                    latents_ref = ref_intermediate_latents[-1 - i]
                    _, latents_cur = latents.chunk(2)
                    latents = torch.cat([latents_ref, latents_cur])

                if guidance_scale > 1.:
                    # added by kookie 24.02.20
                    if mode == 'FlexiEdit' and i == 0:
                        # model_inputs = torch.cat([latents, latent_mask]) 
                        # latents = torch.cat([latents.chunk(2)[1], latent_mask.chunk(2)[1]])
                        # model_inputs = torch.cat([latents] * 2)
                        # model_inputs = torch.cat([latent_sum, latents])
                        # latents = torch.cat([latent_sum.chunk(2)[1], latents.chunk(2)[1]]) 
                        # start_point, end_point = self.points
                        # latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, start_point, end_point)
                        # comb = torch.cat([latent_mask_h[:1], latent_mask_h[:1]])                        
                        # comb = torch.cat([latent_low_alpha_high[:1], latent_low_alpha_high[:1]]) 
                        if self.auto_mask != None:
                            latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, "auto_mask", None, self.auto_mask) # mask_mode, user_mask, auto_mask
                        else:
                            latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, "user_mask", self.user_mask, None)
                        comb = latents
                        model_inputs = torch.cat([comb] * 2)
                        # comb = torch.cat([latent_mask_h]*2)
                        # model_inputs = torch.cat([comb]*2)
            
                    elif mode == 'REINVERSION' and i == 0:
                        model_inputs = torch.cat([latent_sum, latents])
                        latents = torch.cat([latent_sum.chunk(2)[1], latents.chunk(2)[1]]) # [inv_latent, reinv_latent] [(2,4,64,64), (2,4,64,64)] 
                        
                    else:
                        model_inputs = torch.cat([latents] * 2) # [4, 4, 64, 64]
                else:
                    model_inputs = latents
                    
                if unconditioning is not None and isinstance(unconditioning, list):
                    _, text_embeddings = text_embeddings.chunk(2)
                    text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
                
                # NOTE: refined ddim latent in denoising step 0~20
                if mode == 'FlexiEdit' and 0 < i and i <= self.refined_step:
                    start_point, end_point = self.points
                    # latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, start_point, end_point)
                    comb = torch.cat([latent_mask_h]*2)
                    model_inputs = torch.cat([comb] * 2)
                    
                # NOTE: refined ddim latent in denoising step 0~20
                if mode == 'hpf' and 0 < i and i <= self.refined_step:
                    start_point, end_point = self.points
                    # latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, start_point, end_point)
                    comb = torch.cat([latent_low_alpha_high[:1], latent_low_alpha_high[:1]])
                    model_inputs = torch.cat([comb] * 2)
                    
                if mode == 'lpf' and 0 < i and i <= self.refined_step:
                    start_point, end_point = self.points
                    # latent_lpf, latent_hpf, latent_sum, latent_mask_h, latent_mask_l, latent_low_alpha_high, latent_high_alpha_low = self.freq_exp(latents, start_point, end_point)
                    comb = torch.cat([latent_high_alpha_low[:1], latent_high_alpha_low[:1]])
                    model_inputs = torch.cat([comb] * 2)
                
                # predict the noise
                if npi_step >= 0 and i < npi_step:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings_null).sample
            
                else:
                    noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
                # if guidance_scale > 1.:
                #     noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                #     noise_pred = noise_pred_uncon + guidance_scale_batch * (noise_pred_con - noise_pred_uncon)
                
                # do CFG separately for source and target
                # guidance_scale_0 = 1, guidance_scale_1 = 7.5
                if do_separate_cfg:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred_0 = noise_pred_uncon[:batch_size//2,...] + guidance_scale_0 * (noise_pred_con[:batch_size//2,...] - noise_pred_uncon[:batch_size//2,...])                
                    score_delta = noise_pred_con[batch_size//2:,...] - noise_pred_uncon[batch_size//2:,...]
                    if (i >= prox_step) and (prox == 'l0' or prox == 'l1'):
                        if quantile > 0: 
                            threshold = score_delta.abs().quantile(quantile)
                        else:
                            threshold = -quantile  # if quantile is negative, use it as a fixed threshold
                        score_delta -= score_delta.clamp(-threshold, threshold)  # hard thresholding
                        if prox == 'l1':
                            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta) # torch.where(조건, A, B) => 조건에 맞으면 A, 틀리면 B
                            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                    noise_pred_1 = noise_pred_uncon[batch_size//2:,...] + guidance_scale_1 * score_delta
                    noise_pred = torch.cat([noise_pred_0, noise_pred_1], dim=0)
                else:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                #NOTE: added by kookie 24.07.13 
                if self.callback is not None:
                    self.callback(i, t, latents)

                # compute the previous noise sample x_t -> x_t-1
                latents, pred_x0 = self.step(noise_pred, t, latents)
                latents_list.append(latents)
                pred_x0_list.append(pred_x0)

        elif mode == "new_FlexiEdit":
            # set scale
            guidance_scale = self.scale[1]
            self.scale = torch.Tensor(self.scale).reshape(-1, 1, 1, 1).to(DEVICE)

            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
                # using intermediate latents for better reconstruction
                if self.ref_intermediate_latents_app is not None or self.ref_intermediate_latents_struct is not None:
                    latents_app, latents_cur, latents_struct = latents.chunk(3)
                    latents_ref_app = self.ref_intermediate_latents_app[-1 - i] if self.ref_intermediate_latents_app is not None else latents_app
                    latents_ref_struct = self.ref_intermediate_latents_struct[-1 - i] if self.ref_intermediate_latents_struct is not None else latents_struct
                    latents = torch.cat([latents_ref_app, latents_cur, latents_ref_struct])

                if guidance_scale > 1.:
                    model_inputs = torch.cat([latents] * 2)
                else:
                    model_inputs = latents
                
                # register time for feature injection
                register_time(self, t.item())
                
                # predict the noise
                noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
                
                # classifier-free guidance
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + self.scale * (noise_pred_con - noise_pred_uncon)

                # compute the previous noise sample x_t -> x_t-1
                latents, pred_x0 = self.step(noise_pred, t, latents)
                latents_list.append(latents)
                pred_x0_list.append(pred_x0)

                # call the callback, if provided
                if self.callback is not None:
                    self.callback(i, t, latents)

                # latents blend
                if i < self.latent_blend_step and self.latent_blend_type != "non":
                    assert self.image_struct_mask_64 is not None, "latent blend only used when the struct mask exists"
                    # preserve the backgroud of source image
                    if self.latent_blend_type == "bg":
                        # latents[1] = latents[2] * (1 - self.image_struct_mask_64) + latents[1] * self.image_struct_mask_64
                        latents[1] = latents[0] * (1 - self.image_app_mask_64) + latents[1] * self.image_app_mask_64
                        # latents[1] = latents[0] * (1 - self.image_struct_mask_64) * (1 - self.image_app_mask_64) + latents[1] * self.image_struct_mask_64
                    # preserve the foreground of source image
                    elif self.latent_blend_type == "fg":
                        latents[1] = latents[2] * self.image_struct_mask_64 + latents[1] * (1 - self.image_struct_mask_64)

        image = self.latent2image(latents, return_type="pt")
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            latents_list = [self.latent2image(img, return_type="pt") for img in latents_list]
            return image, pred_x0_list, latents_list
        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps,
        reinversion_steps,
        cuda_device,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        mode=None,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        self.cuda_device = cuda_device
        DEVICE = torch.device(self.cuda_device) if torch.cuda.is_available() else torch.device("cpu") 
        if type(image) is Image:
            batch_size = 1
        else:
            batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        overwatch.info(f"text embeddings shape after text_encoder: [bold]{text_embeddings.shape}", ctx_level=1)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        overwatch.info(f"latent shape: [bold]{latents.shape}", ctx_level=1)
        
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        
        overwatch.info(f"Valid timesteps: [bold]{reversed(self.scheduler.timesteps)}", ctx_level=1)
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents]
        pred_x0_list = [latents]
        
        if mode == "REINVERSION":
            self.scheduler.set_timesteps(50)
            self.scheduler.timesteps = self.scheduler.timesteps[50 - reinversion_steps:]
        
        overwatch.info(f"[Start] ==>> DDIM Inversion", ctx_level=1)
        for i, t in enumerate(tqdm(reversed(self.scheduler.timesteps), desc="DDIM Inversion")):
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            # predict the noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            # compute the previous noise sample x_t-1 -> x_t
            latents, pred_x0 = self.next_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list
        return latents, start_latents
