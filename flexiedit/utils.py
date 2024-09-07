import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import make_grid
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import math

@torch.no_grad()
def tensor2numpy(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    # fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list) => Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object) => A filename or a file object
        format(Optional) =>  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # ndarr = tensor.permute(1, 2, 0).to("cpu", torch.uint8).numpy()*255
    # im = Image.fromarray(ndarr)
    # im.save(fp, format=format)
    
    # Handling a list of tensors by stacking
    # if isinstance(tensor, list):
    #     tensor = torch.stack(tensor, dim=0)
    
    # # Convert tensor to CPU and to 'uint8' if not already
    # if tensor.dtype != torch.uint8:
    #     tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    
    # ndarr = tensor.to('cpu').numpy()
    
    # # Adjusting dimensions if necessary (for a single image)
    # if ndarr.ndim == 3:
    #     ndarr = ndarr.transpose(1, 2, 0)
    
    return ndarr


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 0, #2, #NOTE: changed by kookie
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list) => 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional) => Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default => ``8``.
        padding (int, optional) => amount of padding. Default => ``2``.
        normalize (bool, optional) => If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default => ``False``.
        value_range (tuple, optional) => tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional) => If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default => ``False``.
        pad_value (float, optional) => Value for the padded pixels. Default => ``0``.

    Returns:
        grid (Tensor) => the tensor containing grid of images.
    """
    
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def add_text_to_image(image_array, text):
    """
    Function to add "t=10" text to an image array.
    
    :param image_array: Numpy image array in the shape of (height, width, channels).
    :return: Numpy array of the image with text added.
    """
    image = Image.fromarray(image_array)
    
    # Create ImageDraw object
    draw = ImageDraw.Draw(image)
    
    font_path = f'font/Arial Bold.ttf'
    font_size = 32
    
    # font setting
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        # Use default font which may have limited size adjustments
        font = ImageFont.load_default()
        
    # Calculate text position (center top of the image)
    # text = f"t={timestep}"
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (image.width - text_width) / 2
    text_y = 0  # 상단
    
    # Draw text (in red)
    draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
    
    return np.array(image)

def draw_mask(input_image, start_point, end_point):
    
    h, w, c = input_image.shape # (512, 1536, 3)
    image = Image.fromarray(input_image[:, 0:h, :3])
    
    # Create ImageDraw object
    draw = ImageDraw.Draw(image)

    # Draw bounding box (only the border, no fill)
    box_color = (255, 0, 0)  # Red
    thickness = 2  # Border thickness

    # Reconstruct bounding box coordinates from start_point and end_point
    bbox = [start_point[0], start_point[1], end_point[0], end_point[1]]

    # Draw border
    draw.rectangle(bbox, outline=box_color, width=thickness)

    input_image[:, 0:h, :3] = np.array(image)
    return input_image

def slerp(val, low, high):
    """ 
    taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def slerp_tensor(val, low, high):
    """ 
    used in negtive prompt inversion
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device) # batch_size=2
    return latent, latents # latent.shape = (1, 4, 64, 64), latents.shape = (2, 4, 64, 64), 


@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = model.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents


def get_word_inds(text: str, word_place: int, tokenizer):
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
    return np.array(out)

def update_alpha_time_word(alpha, bounds, prompt_ind,
                           word_inds=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps,
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

def txt_draw(text, v="top", h="left", target_size=[512,512]): # width, height
    
    if v == "center" and h == "center":
        fig_width, fig_height = target_size
        ratio = fig_width / fig_height
        plt.figure(dpi=300, figsize=(1,1/ratio))
        plt.text(0.5, 0.5, text, fontsize=3.5, wrap=True, verticalalignment="center", horizontalalignment="center")
    else:
        plt.figure(dpi=300, figsize=(1,1))
        plt.text(-0.1, 1.1, text, fontsize=3.5, wrap=True, verticalalignment=v, horizontalalignment=h)
        
    plt.axis('off')
    
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = image.resize(target_size,Image.ANTIALIAS)
    image = np.asarray(image)[:,:,:3]
    
    plt.close('all')
    
    return image
