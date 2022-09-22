'''TRABAJO 2/3 DINÁMICA DE ESTRUCTURAS
Presentado por: Nelson Esteban Hernández Soto
'''
#%% INICIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_asignado = 7
g = 9.8 # m/s²
# Se asume que se tiene un edificio de concreto reforzado
fpc = 210 # kgf/cm²
E_c = 15100*np.sqrt(fpc)*g*(100**2)/1000  # [kN/m²]
N_PISOS = 3
#%% CONSTANTES PARA MEJORAR LA LECTURA DEL CÓDIGO
NL1, NL2, TIPO = 0, 1, 2     # Nodo local 1, Nodo local 2
X, Y, TH       = 0, 1, 2     # TH = theta
b, h           = 0, 1        # b = base, h = altura

#%% Se define la estructura
xnod = np.array([[0, 0],   # coordenadas de cada nodo [x, y]
                 [0, 3.4],
                 [0, 6.8],
                 [0, 10.2],
                 [6, 0],
                 [6, 3.4],
                 [6, 6.8],
                 [6, 10.2],
                 [12, 0],
                 [12, 3.4],
                 [12, 6.8],
                 [12, 10.2]])

# LaG: local a global: matriz que relaciona nodos locales y globales
# fila = barra
# col1 = nodo global asociado a nodo local 1
# col2 = nodo global asociado a nodo local 2
# (se lee la barra x va del nodo i al nodo j)

#                  NL1   NL2  TIPO -> 1 COL, 2 VIG
barra = np.array([[1,    2,   1],
                  [2,    3,   1],
                  [3,    4,   1],
                  [5,    6,   1],
                  [6,    7,   1],
                  [7,    8,   1],
                  [9,   10,   1],
                  [10,  11,   1],
                  [11,  12,   1],
                  [2,    6,   2],
                  [3,    7,   2],
                  [4,    8,   2],
                  [6,   10,   2],
                  [7,   11,   2],
                  [8,   12,   2]])-1

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
# Dimensiones de los elementos del pórtico np.array([b, h])
COL = np.array([0.40, 0.40])
VIG = np.array([0.30, 0.40])

#                  área       inercias_y       módulo de elasticidad
#                  A(m^2)        I(m^4)           E(kN/m^2)
props = np.array([[COL[b]*COL[h],   (COL[b]*COL[h]**3)/12,     E_c],
                  [VIG[b]*VIG[h],   (VIG[b]*VIG[h]**3)/12,     E_c]])

A = props[:,0];   I = props[:,1];   E = props[:,2]

#%% gdl: grados de libertad
# fila = nodo
# col1 = gdl en dirección x
# col2 = gdl en dirección y
# col3 = gdl en dirección angular antihoraria
gdl = np.arange(ngdl).reshape(nno, 3)  # nodos vs grados de libertad

#%% Ángulo, área y longitud por barra
area = np.zeros(nbar)
ang = np.zeros(nbar)
long = np.zeros(nbar)
for e in range(nbar):
   long[e] = np.hypot(x2[e]-x1[e], y2[e]-y1[e]) # m 
   ang[e] = np.arctan2(y2[e]-y1[e], x2[e]-x1[e]) # radianes
   area[e] = A[barra[e,TIPO]]          # m3

#%% Se dibuja la estructura junto con su numeración
plt.figure(1)
for e in range(nbar):
   plt.plot(xnod[LaG[e,:],X], xnod[LaG[e,:],Y], 'b-')
   
   # Calculo la posición del centro de gravedad de la barra
   cgx = (xnod[LaG[e,NL1],X] + xnod[LaG[e,NL2],X])/2
   cgy = (xnod[LaG[e,NL1],Y] + xnod[LaG[e,NL2],Y])/2
   plt.text(cgx, cgy, str(e+1), color='red')

plt.plot(xnod[:,X], xnod[:,Y], 'ro')
for n in range(nno):
    plt.text(xnod[n,X], xnod[n,Y], str(n+1))
    
plt.axis('equal')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.title('Numeración de la estructura')
plt.show()

#%% ensamblo la matriz de rigidez global
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

#%% grados de libertad del desplazamiento conocidos (c) y desconocidos (d)
apoyos = np.array([[gdl[1-1,X],  0],
                   [gdl[1-1,Y],  0],
                   [gdl[1-1,TH], 0],
                   [gdl[5-1,X],  0],
                   [gdl[5-1,Y],  0],
                   [gdl[5-1,TH], 0],
                   [gdl[9-1,X],  0],
                   [gdl[9-1,Y],  0],
                   [gdl[9-1,TH], 0]])

c = apoyos[:,0].astype(int)
d = np.setdiff1d(np.arange(ngdl), c)
gdl_nc = np.setdiff1d(d, np.union1d(gdl[:,Y], gdl[:,TH]))
gdl_c = np.setdiff1d(d, gdl_nc)

K0 = K[ np.ix_( gdl_nc, gdl_nc )]
K1 = K[ np.ix_( gdl_nc, gdl_c ) ] 
K2 = K[ np.ix_( gdl_c, gdl_nc ) ] 
K3 = K[ np.ix_( gdl_c, gdl_c )  ]

K_con = K0 - K1 @ np.linalg.inv(K3) @ K2

# Ahora lo que resta es sumar los grados de libertad por cada piso para así reducir la matriz
# AQUÍ SE EVIDENCIA LA IMPORTANCIA DE NUMERAR EN ORDEN PRIMERO LAS COLUMNAS Y LUEGO LAS VIGAS

# %%
df = pd.DataFrame(K_con)
filepath = 'MatrizK_con.xlsx'
df.to_excel(filepath, index=False)