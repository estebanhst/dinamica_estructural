'''TRABAJO 2/3 DINÁMICA DE ESTRUCTURAS
Presentado por: 
    Nelson Esteban Hernández Soto
    Heidy Paola Rodríguez Quevedo
Sábado 01 de octubre de 2022
Ya se verificó respecto a los códigos de Juan Pablo
Es importante aclarar que la numeración de la matriz de masa y de rigidez es primera fila/columna, piso 1 y así
'''
#%% INICIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

plt.style.use('ggplot')

#%% CONSTANTES PARA MEJORAR LA LECTURA DEL CÓDIGO
NL1, NL2, TIPO = 0, 1, 2     # Nodo local 1, Nodo local 2
X, Y, TH       = 0, 1, 2     # TH = theta
b, h           = 0, 1        # b = base, h = altura

def reducir_matriz(K, n_pisos):
   # Primero se reduce por las filas
   gdl = np.shape(K)[0]
   K_f = np.zeros((n_pisos, gdl))
   for i in range(n_pisos):
      for j in range(int(gdl/n_pisos)):
         K_f[i] += K[i+n_pisos*j]
   # Ahora se reduce por columnas
   K_r = np.zeros((n_pisos, n_pisos))
   for i in range(n_pisos):
      for j in range(int(gdl/n_pisos)):
         K_r[:,i] += K_f[:,i+n_pisos*j]
   return K_r
def espectro_NSR10(param_loc, param_din, h, grafica = False):
    '''
    param_loc -> diccionario
    Aa  :   # A.2.3-2. Coeficiente de aceleración horizontal pico efectiva.
    Av  :   # A.2.3-2. Coeficiente de velocidad horizontal pico efectiva.
    Fa  :   # A.2.4-3. Coeficiente de amplificación de la aceleración para periodos cortos.
    Fv  :   # A.2.4-4. Coeficiente de amplificación de la aceleración en la zona de periodos intermedios.
    I   :   # A.2.5-1. Coeficiente de importancia.
    
    param_din -> diccionario
    Ct  :   # A.2.5-1.  Coeficiente para el cálculo del periodo.
    alfa:   # A.2.5-1.  Exponente para el cálculo del periodo.
    
    grafica : boolean (True or False)
    '''
    Aa, Av, Fa, Fv, I = param_loc.values()
    Ct, alfa = param_din.values()

    # Cálculo de periodos
    T_a = Ct*(h**alfa)  # [s] Periodo fundamental aproximado, A.4.2-3.
    T_0 = 0.1*(Av*Fv)/(Aa*Fa)   # [s] Periodo inicial.
    T_C = 0.48*(Av*Fv)/(Aa*Fa)  # [s] Periodo corto.
    T_L = 2.4*Fv                # [s] Periodo largo.        Figura A.2.6-1.

    # Aceleración para el periodo fundamental aproximado.
    if T_a < T_C:
        S_a = 2.5*Aa*Fa*I
    elif T_a > T_L:
        S_a = 1.2*Av*Fv*T_L*I/T_a**2
    else:
        S_a = 1.2*Av*Fv*I/T_a
    # Valor del exponente
    if T_a <= 0.5:
        k = 1
    elif T_a > 2.5:
        k = 2
    else:
        k = 0.75+0.5*T_a
    resultados = {
        'S_a': S_a,
        'T_a': T_a,
        'k': k,
        'T_0': T_0,
        'T_C': T_C,
        'T_L': T_L
    }
    if grafica:
        # Gráfica del espectro de aceleraciones 
        t_espectro = np.linspace(0, 10, 100)
        sa_espectro = np.zeros_like(t_espectro)
        for t,i in zip(t_espectro,range(len(t_espectro))):
            if t < T_C:
                sa_espectro[i] = 2.5*Aa*Fa*I
            elif t > T_L:
                sa_espectro[i] = 1.2*Av*Fv*T_L*I/t**2
            else:
                sa_espectro[i] = 1.2*Av*Fv*I/t
        plt.figure()
        plt.plot(t_espectro, sa_espectro, '-k')
        plt.plot(T_a, S_a, '.b', ms=10)
        plt.title('Espectro de aceleraciones A.2 NSR-10')
        plt.xlabel(r'$T\quad [s]$')
        plt.ylabel(r'$S_a[g]$')
        plt.text(1.1*T_a, S_a, r"$(T_a,S_a)$", fontsize=10, color='b')
        plt.grid(visible=True)
        plt.show()
    return resultados
def MFHE(masa_pisos, h_acumu, S_a, k):
    masa_total = np.sum(masa_pisos)   # kN
    Vs = masa_total*(S_a)     # kN
    m_h_k = masa_pisos*(h_acumu**k) # kN
    C_vx = m_h_k/np.sum(m_h_k)      # [%]
    F = np.reshape(Vs*C_vx, (len(masa_pisos),1))              # kN
    return F
def matriz_rigidez_portico(datos_portico):
    xnod = datos_portico['xnod']
    barra = datos_portico['barra']
    apoyos = datos_portico['apoyos']
    COL = datos_portico['COL']
    VIG = datos_portico['VIG']
    E_c = datos_portico['E']
    n_porticos = datos_portico['n_porticos']
    LaG = barra[:, [NL1, NL2]]  # local a global
    tipo = barra[:, TIPO]        # material - 1 columna - 2 viga
    nno  = xnod.shape[0] # número de nodos (numero de filas de xnod)
    nbar = LaG.shape[0]  # número de EFs (numero de filas de LaG)
    ngdl = 3*nno         # número de grados de libertad (tres por nodo)
    # coordenadas de los nodos de cada barra
    x1 = np.zeros(nbar) 
    y1 = np.zeros(nbar)
    x2 = np.zeros(nbar)
    y2 = np.zeros(nbar)
    for e in range(nbar):
        x1[e] = xnod[LaG[e,NL1], X];  x2[e] = xnod[LaG[e,NL2], X]
        y1[e] = xnod[LaG[e,NL1], Y];  y2[e] = xnod[LaG[e,NL2], Y]

    #                  área       inercias_y       módulo de elasticidad
    #                  A(m^2)        I(m^4)           E(kN/m^2)
    props = np.array([[COL[b]*COL[h],   (COL[b]*COL[h]**3)/12,     E_c],
                    [VIG[b]*VIG[h],   (VIG[b]*VIG[h]**3)/12,     E_c]])

    A = props[:,0];   I = props[:,1];   E = props[:,2]

    # gdl: grados de libertad
    # fila = nodo
    # col1 = gdl en dirección x
    # col2 = gdl en dirección y
    # col3 = gdl en dirección angular antihoraria
    gdl = np.arange(ngdl).reshape(nno, 3)  # nodos vs grados de libertad

    # Ángulo, área y longitud por barra
    area = np.zeros(nbar)
    ang = np.zeros(nbar)
    long = np.zeros(nbar)
    for e in range(nbar):
        long[e] = np.hypot(x2[e]-x1[e], y2[e]-y1[e]) # m 
        ang[e] = np.arctan2(y2[e]-y1[e], x2[e]-x1[e]) # radianes
        area[e] = A[barra[e,TIPO]]          # m3
    # Se dibuja la estructura junto con su numeración
    plt.figure(1)
    for e in range(nbar):
        plt.plot(xnod[LaG[e,:],X], xnod[LaG[e,:],Y], 'k-')
        # Calculo la posición del centro de gravedad de la barra
        cgx = (xnod[LaG[e,NL1],X] + xnod[LaG[e,NL2],X])/2
        cgy = (xnod[LaG[e,NL1],Y] + xnod[LaG[e,NL2],Y])/2
        plt.text(cgx, cgy, str(e+1), color='red')

        plt.plot(xnod[:,X], xnod[:,Y], 'ro')
        for n in range(nno):
            plt.text(xnod[n,X], xnod[n,Y], str(n+1), color='k')
    plt.axis('equal')
    plt.grid(visible=True, which='both', color='0.65',linestyle='-')
    plt.title('Numeración del pórtico')
    plt.show()

    # ensamblo la matriz de rigidez global
    K   = np.zeros((ngdl,ngdl))  # separo memoria
    Ke  = nbar*[None]
    T   = nbar*[None]
    idx = np.zeros((nbar,6), dtype=int)

    for e in range(nbar): # para cada barra
        # saco los 6 gdls de la barra e
        idx[e] = np.r_[gdl[LaG[e,NL1],:], gdl[LaG[e,NL2],:]]
        L = long[e]
        # matriz de transformación de coordenadas para la barra e
        c = np.cos(ang[e]);   s = np.sin(ang[e])  # seno y coseno de la inclinación
        T[e] = np.array([[ c,  s,  0,  0,  0,  0],
                            [-s,  c,  0,  0,  0,  0],
                            [ 0,  0,  1,  0,  0,  0],
                            [ 0,  0,  0,  c,  s,  0],
                            [ 0,  0,  0, -s,  c,  0],
                            [ 0,  0,  0,  0,  0,  1]])
        # matriz de rigidez local expresada en el sistema de coordenadas locales
        # para la barra e
        AE = A[tipo[e]]*E[tipo[e]];       L2=L**2
        EI = E[tipo[e]]*I[tipo[e]];       L3=L**3
        Kloc = np.array([
                [ AE/L,   0      ,   0      ,  -AE/L,    0      ,   0      ],  
                [ 0   ,  12*EI/L3,   6*EI/L2,   0   ,  -12*EI/L3,   6*EI/L2],
                [ 0   ,   6*EI/L2,   4*EI/L ,   0   ,   -6*EI/L2,   2*EI/L ],
                [-AE/L,   0      ,   0      ,   AE/L,    0      ,   0      ],
                [ 0   , -12*EI/L3,  -6*EI/L2,   0   ,   12*EI/L3,  -6*EI/L2],
                [ 0   ,   6*EI/L2,   2*EI/L ,   0   ,   -6*EI/L2,   4*EI/L ]])

        # matriz de rigidez local en coordenadas globales
        Ke[e] = T[e].T @ Kloc @ T[e]
        K[np.ix_(idx[e],idx[e])] += Ke[e] # ensambla Ke{e} en K global

    # grados de libertad del desplazamiento conocidos (c) y desconocidos (d)
    nodos_apoyos = np.arange(len(xnod))[xnod[:,1]==0]
    c = np.array([gdl[ij] for ij in apoyos])

    #c = apoyos[:,0].astype(int)
    d = np.setdiff1d(np.arange(ngdl), c)
    gdl_nc = np.setdiff1d(d, np.union1d(gdl[:,Y], gdl[:,TH]))
    gdl_c = np.setdiff1d(d, gdl_nc)

    K0 = K[ np.ix_( gdl_nc, gdl_nc )]
    K1 = K[ np.ix_( gdl_nc, gdl_c ) ] 
    K2 = K[ np.ix_( gdl_c, gdl_nc ) ] 
    K3 = K[ np.ix_( gdl_c, gdl_c )  ]

    K_condensada = (K0 - K1 @ np.linalg.inv(K3) @ K2)*n_porticos
    return K_condensada
def modal_analisis(M, K):
    # Variables simbólicas.
    lam, ome, fi, alf = sp.symbols('lambda, omega, Phi, alpha')

    # Polinomio caracterísitco.
    # lam = ome^2
    poli_car = sp.det(sp.Matrix(K - lam*M))

    # Solución de los lambdas,
    lams_u = np.zeros(len(masa_pisos))
    for i in range(len(lams_u)):
        lams_u[i] = float( sp.re( sp.solve( poli_car, lam )[i]))
    lams = np.sort(lams_u)
    # Cálculo de vibraciones, frecuencias y periodos
    wwi = np.sqrt(lams)  # [rad/s]   Vector de frecuencias angulares.       
    Tt  = 2*np.pi/wwi    # [s]       Vector de periodos de la estructura.
    ff  = 1/Tt           # [Hz]      Vector de frecuencias.

    # -----------------------------------------------------------------------------
    print("\nVibraciones, frecuencias y periodos.")
    print(f">>> {sp.pretty(lam)}: Soluciones {sp.pretty(lam)} del polinomio.")
    print(f">>> {sp.pretty(ome)}: Frecuencias angulares {sp.pretty(ome)} de la edificación.")
    print(">>> T: Periodos T de la edificación.")
    print(">>> f: Frecuencias f de la edificación.\n")

    tabla_vfp = pd.DataFrame(
        data = np.c_[lams.round(5), wwi.round(5), Tt.round(5), ff.round(5)],
        index = np.arange(n_pisos)+1,
        columns = [f"{sp.pretty(lam)}", f"{sp.pretty(ome)} [rad/s]", "T [s]", "f [Hz]"]
    )
    print(tabla_vfp)

    # Periodo de Rayleigh
    T_Rayleigh = 2*np.pi*np.sqrt(np.sum(M*(U/100)**2)/np.sum(F*(U/100)))
    print(f'T Rayleigh = {T_Rayleigh:.5f} s')
    # # %%

    # MATRIZ MODAL
    Phi = np.zeros((n_pisos, n_pisos))

    for j in range(n_pisos):
        # Se calcula el vector de amplitudes del movimiento armónico
        Phi_j = np.linalg.eigh(K-lams[j]*M)[1][:,j]
        # Norma respecto a la masa
        r_j = Phi_j.T @ M @ Phi_j
        # Se agrega el vector normalizado en la matriz modal
        Phi[:,j] = Phi_j/np.sqrt(r_j)

    print('MATRIZ MODAL')
    print(Phi)
    compr_lams = Phi.T @ K @ Phi
    compr_Id = Phi.T @ M @ Phi
    print('Comprobación frecuencias')
    print(compr_lams.round(3))
    print('Comprobación identidad')
    print(compr_Id.round(3))

    # PARTICIPACIÓN MODAL
    alfa = Phi.T @ M @ np.ones((n_pisos,1))
    M_mod_efectiva = alfa**2
    participacion_masa = M_mod_efectiva/np.sum(M_mod_efectiva)
    pctj_participacion = participacion_masa*100


    print('Masa modal efectiva')
    print(M_mod_efectiva.round(2))
    print('Porcentaje de participación de masa')
    print(pctj_participacion.round(2))

    # GRÁFICO DE LOS MODOS Y DESPLAZAMIENTOS
    graf_Phi = np.concatenate((np.zeros((1,n_pisos)),Phi))
    graf_estruct = np.insert(h_acumu,0,0)
    graf_U = np.insert(U,0,0)
    graf_Urel = np.insert(U_rel,0,0)

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    fig.supylabel('Altura [m]')
    fig.suptitle('Desplazamientos y derivas')
    ax = fig.add_subplot(1,2,1)
    ax.grid(visible=True)
    ax.plot(graf_U, graf_estruct, '--b')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, 'og')
    ax.plot(0,0,'_k', markersize=100)
    ax.set_title(f"Desplazamientos [cm]")
    ax = fig.add_subplot(1,2,2)
    ax.grid(visible=True)
    ax.plot(graf_Urel, graf_estruct, ':r')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, 'og')
    ax.plot(np.zeros(n_pisos+1)+DeltaMAX, graf_estruct, '-.g', label=r'$\Delta_{max}$')
    ax.plot(0,0,'_k', markersize=100)
    ax.legend()
    ax.set_title(f"Derivas [cm]")
    plt.show()

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    fig.supylabel('Altura [m]')
    for i in range(n_pisos):
        ax = fig.add_subplot(1,3,1+i)
        ax.grid(visible=True)
        ax.plot(graf_Phi[:,i], graf_estruct, ':b')
        ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
        ax.plot(np.zeros(n_pisos+1), graf_estruct, 'og')
        ax.plot(0,0,'_k', markersize=100)
        ax.set_title(f"Modo {i+1}: "+r"$\omega$"+f"{i+1} = {wwi[i].round(3)} [rad/s]")
        lim = np.max(abs(Phi))+0.05
        ax.set_xlim([-lim, lim])
    plt.show()
    return Phi

# %% INICIO
g = 9.8 # m/s²
# Se asume que se tiene un edificio de concreto reforzado
fpc = 21 # MPa
E_c = 4700*np.sqrt(fpc)*1000  # [kN/m²]
n_pisos = 6
h_entrepiso = 3.5
D_losa = 460                # kgf/m²
D_cub = D_losa/2            # kgf/m²
Lu_dirY = np.array([7,7])         # [m] Luces en dirección Y
Lu_dirX = np.array([9,9,9]) # [m] Luces en dirección X
LY = np.sum(Lu_dirY) # [m] Longitud total de la losa en dirección X
LX = np.sum(Lu_dirX) # [m] Longitud total de la losa en dirección Y
A_losa = LX*LY       # [m²]
# Derivas entre el 0.95% y el 1%
DeltaMAX = 0.01*h_entrepiso*100  # [cm]
DeltaMAXmin = 0.0095*h_entrepiso*100 # [cm]

# Dimensiones de los elementos del pórtico np.array([b, h])
# LA BASE DE LA COLUMNA SE ALINEA CON EL EJE X DE LA PLANTA
COL = np.array([0.85, 0.7])
VIG = np.array([0.4, 0.6])

# Ciudad = Manizales, Grupo de uso = I, Suelo = D
param_loc = {
    'Aa':0.25,
    'Av':0.25,
    'Fa': 1.3,
    'Fv': 1.9,
    'I': 1.0
}
# Parámetros dinámicos para una estructura con pórticos de concreto. 
param_din = {
    'Ct': 0.047,
    'alfa': 0.90
}
#%% ESPECTRO
# [Sa, T_a, k, T_0, T_C, T_L]
espectro_norma_x = espectro_NSR10(param_loc, param_din, h_entrepiso*n_pisos, True)
T_a = espectro_norma_x['T_a']
S_a = espectro_norma_x['S_a']
k = espectro_norma_x['k']
print(f'Propiedades mecánicas del material:\n\
   E= {E_c:.3f} kN/m²')
print(f'Cargas muertas del pórtico:\n\
   D Losa     = {D_losa} kgf/m²\n\
   D Cubierta = {D_cub} kgf/m²\n')
print(f'PERIODO Y ACELERACIÓN DE LA ESTRUCTURA: \n\
   T_a = {T_a:.5f} s\n\
   S_a = {S_a:.5f}(g)\n')

#%% FUERZA HORIZONTAL EQUIVALENTE
masa_pisos = np.array([])
h_acumu = np.array([])
for i in range(1,n_pisos+1):
   h_acumu = np.append(h_acumu, i*h_entrepiso)
   if i < n_pisos:
      masa_pisos = np.append(masa_pisos, D_losa)
   else:
      masa_pisos = np.append(masa_pisos, D_cub)
masa_pisos = masa_pisos*A_losa*(g/1000)    # kN
FHE_piso = MFHE(masa_pisos, h_acumu, S_a, k)           # kN
print('MÉTODO DE LA FUERZA HORIZONTAL EQUIVALENTE:\n\
(se desprecia la masa aportada por las vigas y columnas)')
print(f'F = \n{np.round(FHE_piso,3)}')
#%% MATRICES RIGIDEZ
datos_portico_X = {
    'xnod' : np.array([
        [0, 0],   # coordenadas de cada nodo [x, y]
        [0, 3.5],
        [0, 7.0],
        [0, 10.5],
        [0, 14],
        [0, 17.5],
        [0, 21.0],
        [9, 0],   # coordenadas de cada nodo [x, y]
        [9, 3.5],
        [9, 7.0],
        [9, 10.5],
        [9, 14],
        [9, 17.5],
        [9, 21.0],
        [18, 0],   # coordenadas de cada nodo [x, y]
        [18, 3.5],
        [18, 7.0],
        [18, 10.5],
        [18, 14],
        [18, 17.5],
        [18, 21.0],
        [27, 0],   # coordenadas de cada nodo [x, y]
        [27, 3.5],
        [27, 7.0],
        [27, 10.5],
        [27, 14],
        [27, 17.5],
        [27, 21.0]]),
    'barra' : np.array([
        # LaG: local a global: matriz que relaciona nodos locales y globales
        # fila = barra
        # col1 = nodo global asociado a nodo local 1
        # col2 = nodo global asociado a nodo local 2
        # (se lee la barra x va del nodo i al nodo j)
        # NL1   NL2  TIPO -> 1 COL, 2 VIG
        [1,    2,   1],
        [2,    3,   1],
        [3,    4,   1],
        [4,    5,   1],
        [5,    6,   1],
        [6,    7,   1],
        [8,    9,   1],
        [9,   10,   1],
        [10,  11,   1],
        [11,  12,   1],
        [12,  13,   1],
        [13,  14,   1],
        [15,  16,   1],
        [16,  17,   1],
        [17,  18,   1],
        [18,  19,   1],
        [19,  20,   1],
        [20,  21,   1],
        [22,  23,   1],
        [23,  24,   1],
        [24,  25,   1],
        [25,  26,   1],
        [26,  27,   1],
        [27,  28,   1],
        [2,    9,   2],
        [3,   10,   2],
        [4,   11,   2],
        [5,   12,   2],
        [6,   13,   2],
        [7,   14,   2],
        [9,   16,   2],
        [10,  17,   2],
        [11,  18,   2],
        [12,  19,   2],
        [13,  20,   2],
        [14,  21,   2],
        [16,  23,   2],
        [17,  24,   2],
        [18,  25,   2],
        [19,  26,   2],
        [20,  27,   2],
        [21,  28,   2]])-1,
    'apoyos' : [
        (1 -1,   X),
        (1 -1,   Y),
        (1 -1,   TH),
        (8 -1,   X),
        (8 -1,   Y),
        (8 -1,   TH),
        (15-1,   X),
        (15-1,   Y),
        (15-1,   TH),
        (22-1,   X),
        (22-1,   Y),
        (22-1,   TH)
    ],
    'COL': np.array([COL[h], COL[b]]), # Para considerar en el momento de inercia, la orientación de la columna
    'VIG': VIG,
    'E'  : E_c,
    'n_porticos': 3
}
datos_portico_Y = {
    'xnod' : np.array([
        [0, 0],   # coordenadas de cada nodo [x, y]
        [0, 3.5],
        [0, 7.0],
        [0, 10.5],
        [0, 14],
        [0, 17.5],
        [0, 21.0],
        [7, 0],   # coordenadas de cada nodo [x, y]
        [7, 3.5],
        [7, 7.0],
        [7, 10.5],
        [7, 14],
        [7, 17.5],
        [7, 21.0],
        [14, 0],   # coordenadas de cada nodo [x, y]
        [14, 3.5],
        [14, 7.0],
        [14, 10.5],
        [14, 14],
        [14, 17.5],
        [14, 21.0]]),
    'barra' : np.array([
        # LaG: local a global: matriz que relaciona nodos locales y globales
        # fila = barra
        # col1 = nodo global asociado a nodo local 1
        # col2 = nodo global asociado a nodo local 2
        # (se lee la barra x va del nodo i al nodo j)
        # NL1   NL2  TIPO -> 1 COL, 2 VIG
        [1,    2,   1],
        [2,    3,   1],
        [3,    4,   1],
        [4,    5,   1],
        [5,    6,   1],
        [6,    7,   1],
        [8,    9,   1],
        [9,   10,   1],
        [10,  11,   1],
        [11,  12,   1],
        [12,  13,   1],
        [13,  14,   1],
        [15,  16,   1],
        [16,  17,   1],
        [17,  18,   1],
        [18,  19,   1],
        [19,  20,   1],
        [20,  21,   1],
        [2,    9,   2],
        [3,   10,   2],
        [4,   11,   2],
        [5,   12,   2],
        [6,   13,   2],
        [7,   14,   2],
        [9,   16,   2],
        [10,  17,   2],
        [11,  18,   2],
        [12,  19,   2],
        [13,  20,   2],
        [14,  21,   2]])-1,
    'apoyos' : [
        (1 -1,   X),
        (1 -1,   Y),
        (1 -1,   TH),
        (8 -1,   X),
        (8 -1,   Y),
        (8 -1,   TH),
        (15-1,   X),
        (15-1,   Y),
        (15-1,   TH)
    ],
    'COL': COL,
    'VIG': VIG,
    'E'  : E_c,
    'n_porticos': 4
}

# Ahora lo que resta es sumar los grados de libertad por cada piso para así reducir la matriz
# AQUÍ SE EVIDENCIA LA IMPORTANCIA DE NUMERAR EN ORDEN PRIMERO LAS COLUMNAS Y LUEGO LAS VIGAS
K_c_x = reducir_matriz(matriz_rigidez_portico(datos_portico_X), n_pisos) # kN/m
K_c_y = reducir_matriz(matriz_rigidez_portico(datos_portico_Y), n_pisos) # kN/m
np.savetxt("K_c_x.csv", K_c_x)
np.savetxt("K_c_y.csv", K_c_y)
print('Sección columnas (b h):', COL,'\nSección vigas (b h):', VIG)
print('MATRIZ DE RIGIDEZ CONDENSADA:')
print('>>> K_c_x:')
print(K_c_x.round(2))
print('>>> K_c_y:')
print(K_c_y.round(2))
# Traigo las fuerzas del método de la fuerza horizontal equivalente
# Cambio de unidades de kgf a kN
F = FHE_piso                        # [kN]
U_x = np.linalg.solve(K_c_x, F)*100 # [cm]
U_y = np.linalg.solve(K_c_y, F)*100 # [cm]
U_rel_x = U_x-np.vstack((0,U_x[:-1]))
U_rel_y = U_y-np.vstack((0,U_y[:-1]))
derivas_x = U_rel_x/h_entrepiso
derivas_y = U_rel_y/h_entrepiso

if DeltaMAXmin < max(U_rel_x) < DeltaMAX:
   print("CUMPLE EN X")
else:
   print("NO CUMPLE EN X")
if DeltaMAXmin < max(U_rel_y) < DeltaMAX:
   print("CUMPLE EN Y")
else:
   print("NO CUMPLE EN Y")

tabla_u_derivas = pd.DataFrame(
    data = np.c_[U_x.round(3), U_rel_x.round(3), derivas_x.round(2), 
            U_y.round(3), U_rel_y.round(3), derivas_y.round(2)],
    index = np.arange(n_pisos)+1,
    columns = ["U x [cm]", r"$\Delta$ x [cm]", r"$\delta$ x [%]", "U y [cm]", r"$\Delta$ y [cm]", r"$\delta$ y [%]"]
)
print(tabla_u_derivas)
#%%
# GRÁFICO DE DESPLAZAMIENTOS Y DERIVAS
graf_estruct = np.insert(h_acumu,0,0)
graf_U = np.vstack(
    [np.zeros(2), np.c_[U_x, U_y]])
graf_Urel = np.vstack(
    [np.zeros(2), np.c_[U_rel_x, U_rel_y]])

fig = plt.figure()
fig.set_size_inches(10, 8)
fig.supylabel('Altura [m]')
fig.suptitle('Desplazamientos y derivas')
for i in range(2):
    dir = "x" if i==0 else "y"
    ax = fig.add_subplot(2,2,2*i+1)
    ax.grid(visible=True)
    ax.plot(graf_U[:,i], graf_estruct, '--b')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
    ax.plot(np.zeros(n_pisos+1)[1:], graf_estruct[1:], 'og')
    ax.plot(0,0,'_k', markersize=100)
    ax.set_title(f"Desplazamientos {dir} [cm]")
    ax = fig.add_subplot(2,2,2*i+2)
    ax.grid(visible=True)
    ax.plot(graf_Urel[:,i], graf_estruct, ':r')
    ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
    ax.plot(np.zeros(n_pisos+1)[1:], graf_estruct[1:], 'og')
    ax.plot(np.zeros(n_pisos+1)+DeltaMAX, graf_estruct, '-.g', label=r'$\Delta_{max}$')
    ax.plot(0,0,'_k', markersize=100)
    ax.legend()
    ax.set_title(f"Derivas {dir} [cm]")
plt.show()

fig = plt.figure()
fig.set_size_inches(10, 8)
fig.supylabel('Altura [m]')
fig.suptitle('Desplazamientos y derivas')

ax = fig.add_subplot(1,2,1)
ax.grid(visible=True)
ax.plot(graf_U[:,0], graf_estruct, '--', label='U x')
ax.plot(graf_U[:,1], graf_estruct, '--', label='U y')
ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
ax.plot(np.zeros(n_pisos+1)[1:], graf_estruct[1:], 'og')
ax.plot(0,0,'_k', markersize=100)
ax.legend()
ax.set_title("Desplazamientos [cm]")

ax = fig.add_subplot(1,2,2)
ax.grid(visible=True)
ax.plot(graf_Urel[:,0], graf_estruct, ':', label=r'$\Delta$ x')
ax.plot(graf_Urel[:,1], graf_estruct, ':', label=r'$\Delta$ y')
ax.plot(np.zeros(n_pisos+1), graf_estruct, '-k')
ax.plot(np.zeros(n_pisos+1)[1:], graf_estruct[1:], 'og')
ax.plot(np.zeros(n_pisos+1)+DeltaMAX, graf_estruct, '-.g', label=r'$\Delta_{max}$')
ax.plot(0,0,'_k', markersize=100)
ax.legend()
ax.set_title(f"Derivas [cm]")
plt.show()
# %% SOLUCIÓN MODAL PARA EL CASO NO AMORTIGUADO
M = np.diag(masa_pisos/g) # kN*s²/m
#Phi = modal_analisis(M, K_c_x)

# df1 = pd.DataFrame(K_c)
# df2 = pd.DataFrame(np.stack((U,U_rel)))
# df3 = pd.DataFrame(wwi)
# df4 = pd.DataFrame(Phi)
# df5 = pd.DataFrame(M_mod_efectiva)
# filepath = 'Datos.xlsx'
# with pd.ExcelWriter(filepath, mode='a') as writer:
#    df1.to_excel(writer, "Matriz de rigidez", index=False)
#    df2.to_excel(writer, "Desplazamientos y derivas", index=False)
#    df3.to_excel(writer, "Frecuencias", index=False)
#    df4.to_excel(writer, "Matriz modal", index=False)
#    df5.to_excel(writer, "Masa efectiva", index=False)
