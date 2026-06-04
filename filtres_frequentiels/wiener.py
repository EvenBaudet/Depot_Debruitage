import numpy as np
from .laplacien import *
from scipy.ndimage import uniform_filter

def filtre_wiener(Y, k=10, taille_patch=64):

    L_img = Y.copy()

    h, w = L_img.shape
    L_debruitee = np.zeros((h, w))
    compteur_recouvrement = np.zeros((h, w))
    
    spectre_global = np.fft.fft2(L_img)
    spectre_contour, norme_H = filtre_laplacien(spectre_global)
    contour_Y = np.fft.ifft2(spectre_contour)
    m_contour = np.median(contour_Y)
    sigma_contour = 1.4826 * np.median(np.abs(contour_Y - m_contour))
    sigma = sigma_contour / norme_H
    
    pas = taille_patch // 2

    # Découpage et traitement par patchs locaux
    for i in range(0, h - taille_patch + 1, pas):
        for j in range(0, w - taille_patch + 1, pas):
            patch = L_img[i:i+taille_patch, j:j+taille_patch]
            
            # Filtrage de Wiener fréquentiel sur le patch
            spectre_patch = np.fft.fft2(patch)
            n_p, p_p = spectre_patch.shape
            
            P_bruit = (sigma**2) * (n_p * p_p)
            puissance_brute = np.abs(spectre_patch)**2
            puissance_lissee = uniform_filter(puissance_brute, size=k)
            
            P_signal = np.maximum(puissance_lissee - P_bruit, 0)
            W = P_signal / (P_signal + P_bruit + 1e-12)
            
            spectre_patch_debruite = W * spectre_patch
            patch_debruite = np.real(np.fft.ifft2(spectre_patch_debruite))
            
            L_debruitee[i:i+taille_patch, j:j+taille_patch] += patch_debruite
            compteur_recouvrement[i:i+taille_patch, j:j+taille_patch] += 1

    zones_visitees = compteur_recouvrement > 0
    L_debruitee[zones_visitees] /= compteur_recouvrement[zones_visitees]
    L_debruitee[~zones_visitees] = L_img[~zones_visitees]

    return np.clip(L_debruitee, 0, 255)
