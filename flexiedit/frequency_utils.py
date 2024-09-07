import torch
import torch.fft as fft
import math
import pdb

'''This code is from freeinit => https://github.com/TianxingWu/FreeInit'''
def freq_2d(x, LPF, alpha):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    _x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(_x_freq_low, dim=(-3, -2, -1)).real
    x_low_alpha = fft.ifftn(_x_freq_low*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(_x_freq_high, dim=(-3, -2, -1)).real
    x_high_alpha = fft.ifftn(_x_freq_high*alpha, dim=(-3, -2, -1)).real
    
    _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    _x_freq_low_alpha_high = fft.ifftshift(x_freq_low + x_freq_high*alpha, dim=(-3, -2, -1))
    x_low_alpha_high = fft.ifftn(_x_freq_low_alpha_high, dim=(-3, -2, -1)).real
    
    _x_freq_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high, dim=(-3, -2, -1))
    x_high_alpha_low = fft.ifftn(_x_freq_high_alpha_low, dim=(-3, -2, -1)).real

    _x_freq_alpha_high_alpha_low = fft.ifftshift(x_freq_low*alpha + x_freq_high*alpha, dim=(-3, -2, -1))
    x_alpha_high_alpha_low = fft.ifftn(_x_freq_alpha_high_alpha_low, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum, x_low_alpha, x_high_alpha, x_low_alpha_high, x_high_alpha_low, x_alpha_high_alpha_low

def freq_1d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3))
    x_freq = fft.fftshift(x_freq, dim=(-3))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    #x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    #x_freq_mixed = x_freq_low + x_freq_high

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3))
    x_low = fft.ifftn(x_freq_low, dim=(-3)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3))
    x_high = fft.ifftn(x_freq_high, dim=(-3)).real

    return x_low, x_high

def freq_3d(x, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    #noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    #noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    a = 1.0
    b = 0.1
    #x_freq_low = x_freq * (a * LPF + b * HPF)
    #x_freq_high = x_freq * ( (1-a) * LPF + (1-b) * HPF )
    x_freq_low = x_freq * LPF
    x_freq_high = x_freq * HPF
    
    #x_freq_sum = x_freq_low + x_freq_high
    x_freq_sum = x_freq

    # IFFT
    x_freq_low = fft.ifftshift(x_freq_low, dim=(-3, -2, -1))
    x_low = fft.ifftn(x_freq_low, dim=(-3, -2, -1)).real
    
    x_freq_high = fft.ifftshift(x_freq_high, dim=(-3, -2, -1))
    x_high = fft.ifftn(x_freq_high, dim=(-3, -2, -1)).real
    
    x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
    x_sum = fft.ifftn(x_freq_sum, dim=(-3, -2, -1)).real

    return x_low, x_high, x_sum

def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = (noise_freq) * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain
    
    #x_freq_high = x_freq * HPF
    #x_freq_mixed = 1.5*x_freq_low + x_freq_high

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed

def org_freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError

def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    #mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0
    mask[..., crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask
