import numpy as np

def g_1(s, k):
    return 1.0 / (1.0 + (s / (k + 1e-12))**2)

def g_2(s, k):
    return np.exp(-(s / (k + 1e-12))**2)

def grad_N(I, i, j): return I[i-1, j] - I[i, j]
def grad_S(I, i, j): return I[i+1, j] - I[i, j]
def grad_E(I, i, j): return I[i, j+1] - I[i, j]
def grad_W(I, i, j): return I[i, j-1] - I[i, j]

def c_N(I, i, j, g, k): return g(np.abs(grad_N(I, i, j)), k)
def c_S(I, i, j, g, k): return g(np.abs(grad_S(I, i, j)), k)
def c_E(I, i, j, g, k): return g(np.abs(grad_E(I, i, j)), k)
def c_W(I, i, j, g, k): return g(np.abs(grad_W(I, i, j)), k)

def I_next(I, i, j, g, mu, k):
    return I[i, j] + mu * (
        c_N(I, i, j, g, k) * grad_N(I, i, j) + 
        c_S(I, i, j, g, k) * grad_S(I, i, j) + 
        c_E(I, i, j, g, k) * grad_E(I, i, j) + 
        c_W(I, i, j, g, k) * grad_W(I, i, j))

def iteration(I, g, mu, k):
    h, w = I.shape
    I_new = np.copy(I)
    for i in range(1, h-1):
        for j in range(1, w-1):
            I_new[i, j] = I_next(I, i, j, g, mu, k)
    return I_new

def filtre_perona_malik(image_bruite, n_iter=10, g_func=g_2):
    if len(image_bruite.shape) == 3:
        Y_diffusion = 0.299 * image_bruite[:,:,0] + 0.587 * image_bruite[:,:,1] + 0.114 * image_bruite[:,:,2]
    else:
        Y_diffusion = np.copy(image_bruite).astype(np.float64)

    h, w = Y_diffusion.shape
    mu = 0.05  # inférieur a 0.25 sinon diverge, mais j'ai pris en dessous pour que la diffusion soit plus lente

    for _ in range(int(n_iter)):
        dN = Y_diffusion[:-2, 1:-1] - Y_diffusion[1:-1, 1:-1]
        dS = Y_diffusion[2:, 1:-1] - Y_diffusion[1:-1, 1:-1]
        dE = Y_diffusion[1:-1, 2:] - Y_diffusion[1:-1, 1:-1]
        dW = Y_diffusion[1:-1, :-2] - Y_diffusion[1:-1, 1:-1]
        
        tous_les_gradients = np.concatenate([np.abs(dN).ravel(), np.abs(dS).ravel(), np.abs(dE).ravel(), np.abs(dW).ravel()])
        
        k = np.percentile(tous_les_gradients, 90)
        if k < 1e-5: 
            k = 1e-5
            
        Y_diffusion = iteration(Y_diffusion, g_func, mu, k)

    return np.clip(Y_diffusion, 0.0, 1.0)
