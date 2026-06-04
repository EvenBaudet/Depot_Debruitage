import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  

from filtres_frequentiels.butterworth import filtre_butterworth
from filtres_frequentiels.passe_bas import filtre_passe_bas
from filtres_frequentiels.wiener import filtre_wiener
from filtres_edp.filtrage_edp import filtre_perona_malik, g_1, g_2
from filtres_classiques.classique import filtre_bilateral, filtre_gaussien, filtre_moyenneur

from fonctions_annexe import ajouter_bruit_gaussien
from metriques import SSIM, PSNR

os.makedirs("graphiques", exist_ok=True)

# parametres que l'on va étudier
configuration_filtres = {
    'filtre de wiener': {
        'fonction': filtre_wiener,
        'params': {
            'k': np.linspace(1,27,14),                  
            'taille_patch': np.array([2,4,6,8,10,12,14,16,18,20])      
        }
    },
    'filtre gaussien': {
        'fonction': filtre_gaussien,
        'params': {
            'sigma': np.linspace(0.1, 3.5, 15)
        }
    },
    'filtre moyenneur': {
        'fonction': filtre_moyenneur,
        'params': {
            'taille_noyau': np.array([1,2,3,4,5,6,7,8,9]) 
        }
    },
    'filtre passe-bas': {
        'fonction': filtre_passe_bas,
        'params': {
            'fc': np.linspace(1, 100, 20) 
        }
    },
    'filtre bilatéral': {
        'fonction': filtre_bilateral,
        'params': {
            'sigma_d': np.linspace(0.1, 3, 10),   
            'sigma_r': np.linspace(0.1, 3.5,10)
        }
    },
    'filtre de Perona-Malik': {
        'fonction': filtre_perona_malik,
        'params': {
            'g_func': np.array([g_1, g_2]),     
            'n_iter': np.array([1,3,5,6,7,8,9,10,12,14,16])
        }
    },
    'filtre de Butterworth': {
        'fonction': filtre_butterworth,
        'params': {
            'fc': np.linspace(1, 200, 20),   
            'n_ordre': np.array([1, 2, 3, 4,5,6,7,8])  
        }
    }
}

N = int(input('Nombre d\'images à traiter (1-100) : '))
images_dispo = [f for f in os.listdir("base_image") if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
images_a_traiter = images_dispo[:N]

img_originales = []
img_bruites = []

print("Chargement, redimensionnement (128x128) et bruitage des images...")
for nom_f in images_a_traiter:
    chemin = os.path.join("base_image", nom_f)
    img = mpimg.imread(chemin).astype(np.float64)

    if img.max() > 1.0: # juste au cas ou
        img = img / 255.0

    # Redimensionnement en 256x256 pour que le temps de calcul soit plus court
    h, w = img.shape[:2]
    indices_y = np.linspace(0, h - 1, 256).astype(int)
    indices_x = np.linspace(0, w - 1, 256).astype(int)
    img_final = img[np.ix_(indices_y, indices_x)]
        
    img_originales.append(img_final)
    img_bruites.append(ajouter_bruit_gaussien(img_final, 0.1))

print('\nDébut de l\'optimisation des filtres :')

for nom_filtre, config in configuration_filtres.items():
    print(f"\nÉvaluation du [{nom_filtre}]...")
    func = config['fonction']
    param_names = list(config['params'].keys())
    
    if len(param_names) == 1:
        p1_vals = config['params'][param_names[0]]
        
        ssim_moyens = []
        psnr_moyens = []
        
        for v1 in p1_vals:
            ssim_run, psnr_run = [], []
            for idx in range(N):
                kwargs = {param_names[0]: v1}
                img_f = func(img_bruites[idx], **kwargs)
                
                ssim_run.append(SSIM(img_originales[idx], img_f))
                psnr_run.append(PSNR(img_originales[idx], img_f))
                
            ssim_moyens.append(np.mean(ssim_run))
            psnr_moyens.append(np.mean(psnr_run))
            
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        
        ax1.plot(p1_vals, ssim_moyens, 'g-', label='SSIM')
        ax2.plot(p1_vals, psnr_moyens, 'b-', label='PSNR')
        
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel('SSIM Moyen', color='g')
        ax2.set_ylabel('PSNR Moyen (dB)', color='b')

        nom_fichier_clean = nom_filtre.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join("graphiques", f"optimisation_2d_{nom_fichier_clean}.png"), dpi=300, bbox_inches='tight')

        idx_best_ssim = np.argmax(ssim_moyens)
        idx_best_psnr = np.argmax(psnr_moyens)
        
        print(f"  [RESULTATS {nom_filtre.upper()}] :")
        print(f"    -> SSIM Max : {ssim_moyens[idx_best_ssim]:.4f} pour {param_names[0]} = {p1_vals[idx_best_ssim]}")
        print(f"    -> PSNR Max : {psnr_moyens[idx_best_psnr]:.2f} dB pour {param_names[0]} = {p1_vals[idx_best_psnr]}")

        plt.title(f'Optimisation 2D - {nom_filtre}')
        plt.show()

    elif len(param_names) == 2:
        p1_vals = config['params'][param_names[0]]
        p2_vals = config['params'][param_names[1]]
        
        ssim_grille = np.zeros((len(p1_vals), len(p2_vals)))
        psnr_grille = np.zeros((len(p1_vals), len(p2_vals)))
        
        for i, v1 in enumerate(p1_vals):
            for j, v2 in enumerate(p2_vals):
                ssim_run, psnr_run = [], []
                for idx in range(N):
                    kwargs = {param_names[0]: v1, param_names[1]: v2}
                    img_f = func(img_bruites[idx], **kwargs)
                    
                    ssim_run.append(SSIM(img_originales[idx], img_f))
                    psnr_run.append(PSNR(img_originales[idx], img_f))
                
                ssim_grille[i, j] = np.mean(ssim_run)
                psnr_grille[i, j] = np.mean(psnr_run)
        
        p1_plot_vals = range(len(p1_vals)) if callable(p1_vals[0]) else p1_vals
        p2_plot_vals = range(len(p2_vals)) if callable(p2_vals[0]) else p2_vals
        
        P1, P2 = np.meshgrid(p2_plot_vals, p1_plot_vals) 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        cp1 = ax1.contourf(P1, P2, ssim_grille, levels=15, cmap='viridis')
        lignes1 = ax1.contour(P1, P2, ssim_grille, levels=7, colors='black', alpha=0.4, linewidths=0.8)
        ax1.clabel(lignes1, inline=True, fontsize=8, fmt='%.2f') 
        
        ax1.set_xlabel(param_names[1]) 
        ax1.set_ylabel(param_names[0]) 
        ax1.set_title(f'Optimisation SSIM - {nom_filtre}')
        fig.colorbar(cp1, ax=ax1, label='Score SSIM')
        
        cp2 = ax2.contourf(P1, P2, psnr_grille, levels=15, cmap='plasma')
        lignes2 = ax2.contour(P1, P2, psnr_grille, levels=7, colors='black', alpha=0.4, linewidths=0.8)
        ax2.clabel(lignes2, inline=True, fontsize=8, fmt='%.1f') 
        
        ax2.set_xlabel(param_names[1])
        ax2.set_ylabel(param_names[0])
        ax2.set_title(f'Optimisation PSNR (dB) - {nom_filtre}')
        fig.colorbar(cp2, ax=ax2, label='PSNR (dB)')
        
        if callable(p1_vals[0]):
            ax1.set_yticks(range(len(p1_vals)))
            ax1.set_yticklabels([f.__name__ for f in p1_vals])
            ax2.set_yticks(range(len(p1_vals)))
            ax2.set_yticklabels([f.__name__ for f in p1_vals])
            
        if callable(p2_vals[0]):
            ax1.set_xticks(range(len(p2_vals)))
            ax1.set_xticklabels([f.__name__ for f in p2_vals])
            ax2.set_xticks(range(len(p2_vals)))
            ax2.set_xticklabels([f.__name__ for f in p2_vals])
        
        i_ssim, j_ssim = np.unravel_index(np.argmax(ssim_grille), ssim_grille.shape)
        i_psnr, j_psnr = np.unravel_index(np.argmax(psnr_grille), psnr_grille.shape)
        
        best_p1_ssim, best_p2_ssim = p1_vals[i_ssim], p2_vals[j_ssim]
        best_p1_psnr, best_p2_psnr = p1_vals[i_psnr], p2_vals[j_psnr]
        
        name_p1_ssim = best_p1_ssim.__name__ if callable(best_p1_ssim) else best_p1_ssim
        name_p2_ssim = best_p2_ssim.__name__ if callable(best_p2_ssim) else best_p2_ssim
        name_p1_psnr = best_p1_psnr.__name__ if callable(best_p1_psnr) else best_p1_psnr
        name_p2_psnr = best_p2_psnr.__name__ if callable(best_p2_psnr) else best_p2_psnr

        print(f"  [RESULTATS {nom_filtre.upper()}] :")
        print(f"    -> SSIM Max : {ssim_grille[i_ssim, j_ssim]:.4f} pour ({param_names[0]}={name_p1_ssim}, {param_names[1]}={name_p2_ssim})")
        print(f"    -> PSNR Max : {psnr_grille[i_psnr, j_psnr]:.2f} dB pour ({param_names[0]}={name_p1_psnr}, {param_names[1]}={name_p2_psnr})")
        # ---------------------------------------------

        nom_fichier_clean = nom_filtre.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(os.path.join("graphiques", f"optimisation_3d_{nom_fichier_clean}.png"), dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()
