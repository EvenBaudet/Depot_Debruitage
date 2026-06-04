import numpy as np

def vecteur_frequences(N):
    freq = np.zeros(N)
    for i in range(N):
        if i <= N // 2:
            freq[i] = i / N     
        else:
            freq[i] = (i - N) / N 
    return freq

def filtre_laplacien(s_Y):
    n, p = s_Y.shape

    u = vecteur_frequences(n)
    v = vecteur_frequences(p)
    
    U, V = np.meshgrid(u, v, indexing='ij')
    H_laplacien = -4*(np.sin(np.pi * U)**2 + np.sin(np.pi * V)**2)
    norme_H = np.sqrt(np.mean(H_laplacien**2))
    return s_Y * H_laplacien, norme_H
