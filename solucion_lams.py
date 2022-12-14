import numpy as np

def analisis_modal(M, K):
    '''
    M: Matriz de masa
    K: Matriz de rigidez condensada
    '''
    # lam = ome^2
    # ([K]-lam[M]){Phi} = {0}
    # [K]{Phi} = lam[M]{Phi}
    # Premultiplicando por [M]**-1
    # [M]**-1[K]{Phi}=lam[I]{Phi}
    # (([M]**-1[K])-lam[I]){Phi} = {0}
    # Es decir, el problema se reduce a hallar los valores y vectores propios de [M]**-1[K]
    lams, vectores = np.linalg.eig(np.linalg.inv(M) @ K)
    # Se ordena
    idx = np.argsort(lams)
    lams = lams[idx]
    vectores = vectores[:,idx]
    # Cálculo de vibraciones, frecuencias y periodos
    ome_i = np.sqrt(lams)   # [rad/s]   Vector de frecuencias angulares.       
    T_i  = 2*np.pi/ome_i    # [s]       Vector de periodos de la estructura.
    f_i  = 1/T_i            # [Hz]      Vector de frecuencias.

    # MATRIZ MODAL
    Phi = np.zeros_like(K)

    for j in range(N_PISOS):
        # Se calcula el vector de amplitudes del movimiento armónico
        Phi_j = vectores[:,j]
        # Norma respecto a la masa
        r_j = Phi_j.T @ M @ Phi_j
        # Se agrega el vector normalizado en la matriz modal
        Phi[:,j] = Phi_j/np.sqrt(r_j)
    compr_lams = Phi.T @ K @ Phi
    compr_Id = Phi.T @ M @ Phi
    print('MATRIZ MODAL')
    print(Phi)
    
    print('Comprobación frecuencias')
    print(compr_lams.round(3))
    print('Comprobación identidad')
    print(compr_Id.round(3))

    return Phi, ome_i

N_PISOS = 6
h_entrepiso = 3.5 # m
g = 9.8 # m/s²
D_losa = 460                # kgf/m²
D_cub = D_losa/2            # kgf/m²
Lu_dirY = np.array([7,7])         # [m] Luces en dirección Y
Lu_dirX = np.array([9,9,9]) # [m] Luces en dirección X
LY = np.sum(Lu_dirY) # [m] Longitud total de la losa en dirección X
LX = np.sum(Lu_dirX) # [m] Longitud total de la losa en dirección Y
A_losa = LX*LY       # [m²]

masa_pisos = np.ones(N_PISOS)*D_losa
masa_pisos[-1] = D_cub
masa_pisos = masa_pisos*A_losa*(g/1000)    # kN

M = np.diag(masa_pisos/g) # kN*s²/m
K_c_x=np.array([#kN/m
    [4179264.96,	-2577615.836,	943599.7904,	-229480.4966,	53320.14703,	-8660.077601],
    [-2577615.836,	3341402.836,	-2372733.04,	891236.6551,	-207083.1385,	34059.11889],
    [943599.7904,	-2372733.04,	3289019.947,	-2350274.749,	846278.7074,	-138701.5957],
    [-229480.4966,	891236.6551,	-2350274.749,	3244142.889,	-2178182.269,	569683.4377],
    [53320.14703,	-207083.1385,	846278.7074,	-2178182.269,	2542293.635,	-1044303.504],
    [-8660.077601,	34059.11889,	-138701.5957,	569683.4377,	-1044303.504,	585855.581]
    ])
K_c_y = np.array([#kN/m
    [2876620.673,	-1750042.08,	604149.2971,	-138952.0343,	30738.90712,	-4662.510258],
    [-1750042.08,	2333458.59,	-1624644.506,	574052.1374,	-127200.4999,	20714.97241],
    [604149.2971,	-1624644.506,	2303311.618,	-1612722.968,	550093.4223,	-87992.91675],
    [-138952.0343,	574052.1374,	-1612722.968,	2279573.734,	-1516806.403,	384346.9174],
    [30738.90712,	-127200.4999,	550093.4223,	-1516806.403,	1866308.417,	-796286.8211],
    [-4662.510258,	20714.97241,	-87992.91675,	384346.9174,	-796286.8211,	482637.5657]
    ])

print('MATRIZ DE MASA:')
print(np.round(M,1),'\n')
print('MATRIZ DE RIGIDEZ en X:')
print(np.round(K_c_x,1),'\n')
print('MATRIZ DE RIGIDEZ en Y:')
print(np.round(K_c_y,1))
Phi_x, omegas_x = analisis_modal(M, K_c_x)
Phi_y, omegas_y = analisis_modal(M, K_c_y)
print('Fin (?)')
