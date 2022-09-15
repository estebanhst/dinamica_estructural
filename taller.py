# Trabajo 1/3 DINÁMICA DE ESTRUCTURAS
import numpy as np
# DEFINICIÓN DE VARIABLES
g = 9.81  # m/s²
M6 = 2000 # kg/m²
l = 6     # m lado losa
h = 20    # m altura
fpc = 210 # kgf/cm² f'c del concreto

# CONDICIONES INICIALES
x0 = 0.009*h # m Posición inicial impacto natural, dirección x
v0 = 8 # m/s Velocidad inicial, dirección x

# CÁLCULOS
A = l*l   # m² Área losa
m = M6*A  # kg
delta_max = 0.007*h # m Desplazamiento máximo asumido en dirección x
E = 15100*np.sqrt(fpc)*g*(100**2) # N/m² módulo de E. del Concreto NSR-10  

# RIGIDEZ DE LA ESTRUCTURA
# Se toma esta rigidez dado que es la que aportan las columnas de forma
# perpendicular a su eje axial. Se considera una sección cuadrada para las
# columnas.
b = 0.3 # m
I = b*b**3/12 # m**4
k_col = 12*E*I/h**3 # N/m
k_serie = 1/(2*(1/k_col))
k = 2*k_serie # PARALELO

w = np.sqrt(k/m)