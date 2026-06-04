import numpy as np
from scipy.ndimage import uniform_filter

def SSIM(img1, img2, window_size=11):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    c1 = (0.01 * 1.0)**2
    c2 = (0.03 * 1.0)**2
    
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    var1 = uniform_filter(img1**2, size=window_size) - mu1_sq
    var2 = uniform_filter(img2**2, size=window_size) - mu2_sq
    cov12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2
    
    num = (2 * mu1_mu2 + c1) * (2 * cov12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (var1 + var2 + c2)
    
    ssim_map = num / den
    pad = window_size // 2
    return np.mean(ssim_map[pad:-pad, pad:-pad])

def PSNR(Y, Y_filtre):
    Y = Y.astype(np.float64)
    Y_filtre = Y_filtre.astype(np.float64)
    mse = np.mean((Y - Y_filtre) ** 2)

    if mse == 0:
        return 100.0

    L = 1.0
    psnr = 10 * np.log10((L**2) / mse)
    return psnr
