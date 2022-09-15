'''Trabajo 1/3 DINÁMICA DE ESTRUCTURAS
#completar luego
'''
# %% importación de librerías
import numpy as np
import matplotlib.pyplot as plt

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
a_delta_max = 0.95*g # Aceleración asociada al desplazamiento máximo
fs = m*a_delta_max # N Fuerza que produce la máxima deformación

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
b_min = (12*I_min)**(1/4) # m base mínima columna sección cuadrada
# Propiedades colocadas
# b = b_min
b = 0.85 # m
I = b*b**3/12 # m^4
k_col = 12*E*I/h**3 # N/m
k = 4*k_col # cuatro columnas
w = np.sqrt(k/m)

# %% Vectores
delta_t = 0.01; # s
tt = np.arange(0,10, delta_t)
xxi = np.array([0.0, 0.05, 0.10, 0.30])
wD = np.sqrt(1-xxi**2)*w
cc = 2*xxi*m*w


# %% Funciones
def x_t(t, w, xi, v0, x0, wD):
    x = np.exp(-xi*w*t)*(x0*np.cos(wD*t)+(v0+xi*x0*w)/wD*np.sin(wD*t))
    return x
def v_t(t, w, xi, v0, x0, wD):
    const = (xi*w*v0+xi**2*w**2*x0+x0*wD**2)/wD
    v = np.exp(-xi*w*t)*(v0*np.cos(wD*t)+const*np.sin(wD*t))
    return v
def a_t(t, w, xi, v0, x0, wD):
    const = (xi*w*v0+xi**2*w**2*x0+x0*wD**2)/wD
    a = np.exp(-xi*w*t)*(-(xi*w*v0+const*wD)*np.cos(wD*t)+(const*xi*w-v0*wD)*np.sin(wD*t))
    return a

# %% Gráficas
m = len(xxi)
n = len(tt)
xx = np.empty((m,n))
vv = np.empty((m,n))
aa = np.empty((m,n))
# fi = np.empty((m,n))
# fa = np.empty((m,n))
# fr = np.empty((m,n))


# Desplazamientos, velocidades y aceleraciones
for i in range(m):
    xx[i] = x_t(tt, w, xxi[i], v0, x0, wD[i])
    vv[i] = v_t(tt, w, xxi[i], v0, x0, wD[i])
    aa[i] = a_t(tt, w, xxi[i], v0, x0, wD[i])

# Fuerzas de inercia, amortiguamiento y rigidez

fi = m*aa
fa = (np.tile(cc,(1000,1)).T)*vv
fr = k*xx



fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

for i in range(m):
    ax1.plot(tt,xx[i], label=r'$\xi=$'+ str(xxi[i]))
    ax2.plot(tt,vv[i], label=r'$\xi=$'+ str(xxi[i]))
    ax3.plot(tt,aa[i], label=r'$\xi=$'+ str(xxi[i]))

ax1.set_title("Desplazamientos")
ax1.set_ylabel(r'$x(t) [m]$')
ax1.set_xlabel(r'$t [s]$')
ax1.legend(loc=0)
ax1.grid()
ax2.set_title("Velocidades")
ax2.set_ylabel(r'$v(t) [m/s]$')
ax2.set_xlabel(r'$t [s]$')
ax2.legend(loc=0)
ax2.grid()
ax3.set_title("Aceleraciones")
ax3.set_ylabel(r'$a(t) [m/s^2]$')
ax3.set_xlabel(r'$t [s]$')
ax3.legend(loc=0)
ax3.grid()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

for i in range(m):
    ax1.plot(tt,fr[i], label=r'$\xi=$'+ str(xxi[i]))
    ax2.plot(tt,fa[i], label=r'$\xi=$'+ str(xxi[i]))
    ax3.plot(tt,fi[i], label=r'$\xi=$'+ str(xxi[i]))

ax1.set_title("Fuerzas de rigidez")
ax1.set_ylabel(r'$F_r(t) [N]$')
ax1.set_xlabel(r'$t [s]$')
ax1.legend(loc=0)
ax1.grid()
ax2.set_title("Fuerzas de amortiguamiento")
ax2.set_ylabel(r'$F_a(t) [N]$')
ax2.set_xlabel(r'$t [s]$')
ax2.legend(loc=0)
ax2.grid()
ax3.set_title("Fuerzas de inercia")
ax3.set_ylabel(r'$F_I(t) [N]$')
ax3.set_xlabel(r'$t [s]$')
ax3.legend(loc=0)
ax3.grid()
plt.show()
# %%
