import matplotlib.pyplot as plt
import numpy as np

## Filtre bilatéral
def voisinage_bilateral(rayon, sigma, Y, Y_pad, n, p):
    sigma_r, sigma_d = sigma
    normalisation = np.zeros((n, p))
    wp = np.zeros((n, p))
    k = rayon
    for dk in range(-k, k+1):
        for dl in range(-k, k+1):

            voisin = Y_pad[k+dk : k+dk+n, k+dl : k+dl+p]
            
            g = np.exp(- ((Y - voisin)/sigma_r)**2 /2 - (dk**2 + dl**2) / (2*sigma_d**2))
            
            normalisation += g
            wp += voisin*g
    
    return wp/normalisation

## Filtre gaussien
def voisinage_gaussien(rayon, sigma, Y_pad, n, p):
    normalisation = np.zeros((n, p))
    wp = np.zeros((n, p))
    k = rayon
    for dk in range(-k, k+1):
        for dl in range(-k, k+1):

            voisin = Y_pad[k+dk : k+dk+n, k+dl : k+dl+p]
            
            g = np.exp(- (dk**2 + dl**2) / (2*sigma**2))
            
            normalisation += g
            wp += voisin*g
    
    return wp/normalisation


## Filtre médian
def voisinage_median(i, j, n, p, fenetre, Y):
    k = fenetre[-1]
    region = Y[i - k: i + k + 1, j - k: j + k +1]
    return np.median(region)

## Parcours
def filtre(Y, fenetre, sigma, choix_filtre):
    n,p = len(Y), len(Y[0])
    Y_debruite = np.zeros((n,p))

    rayon = fenetre[-1]
    Y_pad = np.pad(Y, pad_width=rayon, mode='reflect')

    if choix_filtre == 0:
        for i in range(rayon, n + rayon):
            for j in range(rayon, p + rayon):
                Y_debruite[i - rayon, j - rayon] = voisinage_median(i, j, n, p, fenetre, Y_pad)
    elif choix_filtre == 1:
        Y_debruite = voisinage_gaussien(rayon, sigma, Y_pad, n, p)
    elif choix_filtre == 2:
        Y_debruite = voisinage_bilateral(rayon, sigma, Y, Y_pad, n, p)

    return Y_debruite