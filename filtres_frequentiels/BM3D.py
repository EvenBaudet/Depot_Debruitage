import numpy as np
from laplacien import *

# Finalement on ne l'utilise pas car c'est plutot un filtre empirique difficile à étudier d'un point de vue de traitement du signal

PATCH_SIZE    = 8      
SEARCH_WINDOW = 39     
MAX_MATCHES   = 16     
THRESHOLD_3D  = 2.7 / 255.0   

#estimation sigma
def estimation_sigma(spectre_Y):
    spectre_contour_Y, norme_H = filtre_laplacien(spectre_Y)
    contour_Y = np.fft.ifft2(spectre_contour_Y)
    m_contour = np.median(contour_Y)
    sigma_contour = 1.4826 * np.median(np.abs(contour_Y - m_contour))
    SIGMA = sigma_contour / norme_H
    return SIGMA

def dct_matrix(n):
    k = np.arange(n).reshape(-1, 1)      
    i = np.arange(n).reshape(1, -1)     
    D = np.cos(np.pi * k * (2*i + 1) / (2*n))
    D[0, :] /= np.sqrt(n)
    D[1:, :] *= np.sqrt(2.0 / n)
    return D                              

DCT_PATCH = dct_matrix(PATCH_SIZE)       
IDCT_PATCH = DCT_PATCH.T                


def dct2_patch(block):
    return DCT_PATCH @ block @ DCT_PATCH.T


def idct2_patch(block):
    return IDCT_PATCH @ block @ IDCT_PATCH.T


def dct1d_along_axis0(stack):
    n = stack.shape[2]
    D = dct_matrix(n)

    flat = stack.reshape(-1, n)    
    flat_dct = flat @ D.T           
    return flat_dct.reshape(stack.shape)


def idct1d_along_axis0(stack):
    n = stack.shape[2]
    D = dct_matrix(n)
    flat = stack.reshape(-1, n)
    flat_idct = flat @ D            
    return flat_idct.reshape(stack.shape)



def block_matching(image, ref_row, ref_col, patch_size, search_win, max_matches, threshold):
    H, W = image.shape
    P = patch_size

    ref_patch = image[ref_row:ref_row + P, ref_col:ref_col + P]  # (P, P)

    r_min = max(0, ref_row - search_win // 2)
    r_max = min(H - P, ref_row + search_win // 2)
    c_min = max(0, ref_col - search_win // 2)
    c_max = min(W - P, ref_col + search_win // 2)

    rows = np.arange(r_min, r_max + 1)
    cols = np.arange(c_min, c_max + 1)
    R, C = np.meshgrid(rows, cols, indexing='ij')  # (nr, nc)
    R_flat = R.ravel()
    C_flat = C.ravel()

    candidates = np.array([image[r:r+P, c:c+P] for r, c in zip(R_flat, C_flat)])

    dists = np.sum((candidates - ref_patch) ** 2, axis=(1, 2)) / (P * P)

    valid = np.where(dists < threshold)[0]
    valid = valid[np.argsort(dists[valid])][:max_matches]

    return [(R_flat[i], C_flat[i]) for i in valid]


def hard_threshold_3d(group_3d, sigma, patch_size):

    lam = 2.7 
    threshold = lam * sigma

    group_dct = np.stack([dct2_patch(group_3d[:, :, k]) for k in range(group_3d.shape[2])],axis=2)
    group_dct3d = dct1d_along_axis0(group_dct)

    mask = np.abs(group_dct3d) >= threshold
    group_filtered = group_dct3d * mask

    n_nonzero = max(1, np.sum(mask))

    group_idct3d = idct1d_along_axis0(group_filtered)
    group_out = np.stack([idct2_patch(group_idct3d[:, :, k]) for k in range(group_idct3d.shape[2])],axis=2)

    weight = 1.0 / n_nonzero

    return group_out, weight


def aggregate(image_shape, matches, group_out, weight, patch_size):

    H, W = image_shape
    P = patch_size
    numerator   = np.zeros((H, W))
    denominator = np.zeros((H, W))

    for k, (r, c) in enumerate(matches):
        numerator[r:r + P, c:c + P]   += weight * group_out[:, :, k]
        denominator[r:r + P, c:c + P] += weight

    return numerator, denominator


def wiener_filter_3d(group_noisy, group_basic, sigma, patch_size):

    N = group_noisy.shape[2]

    def dct3d(grp):
        d2 = np.stack([dct2_patch(grp[:, :, k]) for k in range(N)], axis=2)
        return dct1d_along_axis0(d2)

    Y_noisy_dct = dct3d(group_noisy)
    Y_basic_dct = dct3d(group_basic)

    power_basic = Y_basic_dct ** 2
    wiener_coef = power_basic / (power_basic + sigma ** 2)

    Y_filtered = wiener_coef * Y_noisy_dct

    weight = 1.0 / (np.sum(wiener_coef ** 2) + 1e-8)

    def idct3d(grp):
        i1 = idct1d_along_axis0(grp)
        return np.stack([idct2_patch(i1[:, :, k]) for k in range(N)], axis=2)

    group_out = idct3d(Y_filtered)

    return group_out, weight



def filtre_bm3d(noisy_image, step=3):

    sigma = estimation_sigma(np.fft.fft2(noisy_image))

    H, W = noisy_image.shape
    P = PATCH_SIZE

    num1  = np.zeros((H, W))
    den1  = np.zeros((H, W))
    masque1 = np.zeros((H, W), dtype=bool)

    for r in range(0, H - P + 1, step):
        for c in range(0, W - P + 1, step):

            matches = block_matching(noisy_image, r, c, P,SEARCH_WINDOW, MAX_MATCHES, THRESHOLD_3D)

            group = np.stack([noisy_image[rr:rr+P, cc:cc+P] for rr, cc in matches],axis=2)

            group_filtered, weight = hard_threshold_3d(group, sigma, P)

            n_, d_ = aggregate((H, W), matches, group_filtered, weight, P)
            num1 += n_
            den1 += d_
            masque1[r:r+P, c:c+P] = True

    basic_estimate = num1 / np.maximum(den1, 1e-8)
    basic_estimate[~masque1] = noisy_image[~masque1]

    num2 = np.zeros((H, W))
    den2 = np.zeros((H, W))
    masque2 = np.zeros((H, W), dtype=bool)

    for r in range(0, H - P + 1, step):
        for c in range(0, W - P + 1, step):

            matches = block_matching(basic_estimate, r, c, P,SEARCH_WINDOW, MAX_MATCHES, THRESHOLD_3D)

            group_noisy = np.stack([noisy_image[rr:rr+P, cc:cc+P] for rr, cc in matches],axis=2)
            group_basic = np.stack([basic_estimate[rr:rr+P, cc:cc+P] for rr, cc in matches],axis=2)

            group_filtered, weight = wiener_filter_3d(group_noisy, group_basic, sigma, P)

            n_, d_ = aggregate((H, W), matches, group_filtered, weight, P)
            num2 += n_
            den2 += d_
            masque2[r:r+P, c:c+P] = True

    final_estimate = num2 / np.maximum(den2, 1e-8)
    final_estimate[~masque2] = basic_estimate[~masque2]

    return np.clip(final_estimate, 0, 1)