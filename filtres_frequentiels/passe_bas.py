import numpy as np

def filtre_passe_bas(Y, fc=30):

    L_img = Y.copy()

    n, p = L_img.shape
    spectre_Y = np.fft.fftshift(np.fft.fft2(L_img))
    
    # Création du masque idéal passe-bas (fréquence de coupure)
    u = np.arange(n) - n // 2
    v = np.arange(p) - p // 2
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    
    H = np.where(D <= fc, 1, 0)
    
    spectre_debruite_Y = spectre_Y * H
    L_debruitee = np.real(np.fft.ifft2(np.fft.ifftshift(spectre_debruite_Y)))

    return np.clip(L_debruitee, 0, 255)