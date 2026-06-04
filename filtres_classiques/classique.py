import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def gaussien(x, sigma):
    return np.e**(-x / (2 * sigma**2))

def debruitage_gaussien(i, j, n, p, fenetre, sigma, Y):
    normalisation, wp = 0, 0
    for k in fenetre:
        for l in fenetre:
            if 0 <= i + k < n and 0 <= j + l < p:
                w = gaussien(k**2 + l**2, sigma)
                normalisation += w
                wp += Y[i + k, j + l] * w
    return wp / normalisation if normalisation != 0 else Y[i, j]

def filtre_gaussien(image_bruite, sigma=1.5):
    taille_fenetre = int(2 * np.ceil(3 * sigma) + 1)
    
    n, p = image_bruite.shape
    Y_debruite = np.zeros((n, p))
    
    r = taille_fenetre // 2
    fenetre = list(range(-r, r + 1))

    for i in range(n):
        for j in range(p):
            Y_debruite[i, j] = debruitage_gaussien(i, j, n, p, fenetre, sigma, image_bruite)
    
    return np.clip(Y_debruite, 0, 255)

def debruitage_bilateral(i, j, n, p, fenetre, sigma_d, sigma_r, Y):
    normalisation, wp = 0, 0
    for k in fenetre:
        for l in fenetre:
            if 0 <= i + k < n and 0 <= j + l < p:
                w = gaussien((Y[i, j] - Y[i + k, j + l])**2, sigma_d) * gaussien(k**2 + l**2, sigma_r)
                normalisation += w
                wp += Y[i + k, j + l] * w
    return wp / normalisation if normalisation != 0 else Y[i, j]

def filtre_bilateral(image_bruite, sigma_d=63.6, sigma_r=1.4, taille_fenetre=5):
    n, p = image_bruite.shape
    Y_debruite = np.zeros((n, p))
    
    r = taille_fenetre // 2
    fenetre = list(range(-r, r + 1))

    for i in range(n):
        for j in range(p):
            Y_debruite[i, j] = debruitage_bilateral(i, j, n, p, fenetre, sigma_d, sigma_r, image_bruite)
    
    return np.clip(Y_debruite, 0, 255)

def debruitage_moyenneur(i, j, n, p, fenetre, Y):
    moy, ct = 0, 0
    for k in fenetre:
        for l in fenetre:
            if 0 <= i + k < n and 0 <= j + l < p:
                moy += Y[i + k, j + l]
                ct += 1
    return moy / ct if ct != 0 else Y[i, j]
    
def filtre_moyenneur(image_bruite, taille_noyau=5):
    n, p = image_bruite.shape
    Y_debruite = np.zeros((n, p))

    r = int(taille_noyau) // 2
    fenetre = list(range(-r, r + 1))

    for i in range(n):
        for j in range(p):
            Y_debruite[i, j] = debruitage_moyenneur(i, j, n, p, fenetre, image_bruite)
        
    return np.clip(Y_debruite, 0, 255)

# les metriques servaient pour l'ancienne étude des hyperparametres mais je les laisse au cas ou
def SSIM(img1, img2, window_size=11):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    n, p = img1.shape
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    
    scores_ssim = []

    for i in range(0, n - window_size + 1, 1):
        for j in range(0, p - window_size + 1, 1):
            
            fenetre1 = img1[i:i+window_size, j:j+window_size]
            fenetre2 = img2[i:i+window_size, j:j+window_size]
            
            mu1 = np.mean(fenetre1)
            mu2 = np.mean(fenetre2)

            var1 = np.var(fenetre1)
            var2 = np.var(fenetre2)
            cov12 = np.mean(fenetre1 * fenetre2) - (mu1 * mu2)
            
            num = (2 * mu1 * mu2 + c1) * (2 * cov12 + c2)
            den = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
            
            ssim_map = num / den
            scores_ssim.append(ssim_map)

    return np.mean(scores_ssim)

def PSNR(Y, Y_filtre):
    Y = Y.astype(np.float64)
    Y_filtre = Y_filtre.astype(np.float64)
    mse = np.mean((Y - Y_filtre) ** 2)

    if mse == 0:
        return 100.0

    L = 255.0
    psnr = 10 * np.log10((L**2) / mse)

    return psnr
