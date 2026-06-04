import numpy as np

def ajouter_bruit_gaussien(image, sigma_bruit=0.1):
    bruit = np.random.normal(0, sigma_bruit, image.shape)
    image_bruitee = image + bruit
    return np.clip(image_bruitee, 0, 1)

def ajouter_bruit_gaussien(image, sigma=0.1): 
    proportion = 0.05 
    img_bruitee = np.copy(image)
    rnd = np.random.rand(*image.shape)
    img_bruitee[rnd < (proportion / 2)] = 1.0
    img_bruitee[(rnd >= (proportion / 2)) & (rnd < proportion)] = 0.0
    return np.clip(img_bruitee, 0.0, 1.0)

def ajouter_bruit_gaussien(image, sigma=0.1):
    peak = 100.0 
    img_norm = np.clip(image, 0.0, 1.0).astype(np.float64)
    img_photons = img_norm * peak
    img_bruitee_photons = np.random.poisson(img_photons).astype(np.float64)
    img_final = img_bruitee_photons / peak
    
    return np.clip(img_final, 0.0, 1.0)