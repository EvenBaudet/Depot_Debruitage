import numpy as np

def filtre_butterworth(Y, fc=30, n_ordre=2):

    L_img = Y.copy()

    n, p = L_img.shape
    spectre_Y = np.fft.fftshift(np.fft.fft2(L_img))
    
    # Création du filtre de Butterworth
    u = np.arange(n) - n // 2
    v = np.arange(p) - p // 2
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    
    # Formule théorique de Butterworth passe-bas
    H = 1 / (1 + (D / (fc + 1e-12))**(2 * n_ordre))
    
    spectre_debruite_Y = spectre_Y * H
    L_debruitee = np.real(np.fft.ifft2(np.fft.ifftshift(spectre_debruite_Y)))

    return np.clip(L_debruitee, 0, 255)