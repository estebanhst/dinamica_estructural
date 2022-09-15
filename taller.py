'''Trabajo 1/3 DINÁMICA DE ESTRUCTURAS
#completar luego
'''
# %% importación de librerías
import numpy as np

# %% definición de variables
g = 9.81  # m/s²
M6 = 2000 # kg/m² carga losa
l = 6     # m lado losa
h = 20    # m altura
fpc = 210 # kgf/cm² f'c del concreto

# %% condiciones iniciales
x0 = 0.009*h # m Posición inicial impacto natural, dirección x
v0 = 8       # m/s Velocidad inicial, dirección x

# %% cálculos
A = l*l   # m² Área losa
m = M6*A  # kg
delta_max = 0.007*h # m Desplazamiento máximo asumido en dirección x
E = 15100*np.sqrt(fpc)*g*(100**2) # N/m² módulo de E. del Concreto NSR-10  
fs = m*0.95*g # N Fuerza que produce la máxima deformación

# %% cálculo de rigidez
# Asumiendo el comportamiento elástico lineal del material, puede determinarse
# una rigidez mínima para las condiciones dadas del problema, bajo el enfoque de fuerza estática 
# equivalente (1.8.1, Chopra)
k_min = fs/delta_max # N/m
# Se toma esta rigidez dado que es la que aportan las columnas de forma
# perpendicular a su eje axial. Se considera una sección cuadrada para las
# columnas.
n_cols = 4 # número de columnas
I_min = k_min*h**3/(12*n_cols*E) # m^4
b_min = (12*I_min)**(1/4) # m
# Propiedades colocadas
# b = b_min
b = 0.85 # m
I = b*b**3/12 # m^4
k_col = 12*E*I/h**3 # N/m
k = 4*k_col # cuatro columnas

w = np.sqrt(k/m)

# %%
