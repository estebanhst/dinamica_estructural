'''Trabajo 1/3 DINÁMICA DE ESTRUCTURAS
Lunes 19 de Septiembre de 2022
Presentado por:
Heidy Paola Rodríguez Quevedo CC 1007584977 hprodriguezqu@unal.edu.co
Nelson Esteban Hernandez Soto CC 1004680056 nhernandez@unal.edu.co

El grupo es el #6
'''
# %% Importación de librerías
import numpy as np
import matplotlib.pyplot as plt

# %% Funciones
def x_t(t, w, xi, v0, x0, wD):
    '''Solución del desplazamiento en el tiempo de una estructura bajo condiciones de 
    vibración libre amortiguada. Ingresar con unidades consistentes.
    t: tiempo [s]
    w: frecuencia natural [1/s]
    xi: c/cc relación entre amortiguamiento y amortiguamiento crítico
    v0: velocidad inicial [m/s]
    x0: posición inicial [m]
    wD: frecuencia amortiguada [1/s]
    '''
    x = np.exp(-xi*w*t)*(x0*np.cos(wD*t)+(v0+xi*x0*w)/wD*np.sin(wD*t))
    return x
def v_t(t, w, xi, v0, x0, wD):
    '''Solución de la velocidad en el tiempo de una estructura bajo condiciones de 
    vibración libre amortiguada. Ingresar con unidades consistentes. (VER FUNC. x_t())
    '''
    const = (xi*w*v0+xi**2*w**2*x0+x0*wD**2)/wD
    v = np.exp(-xi*w*t)*(v0*np.cos(wD*t)-const*np.sin(wD*t))
    return v
def a_t(t, w, xi, v0, x0, wD):
    '''Solución de la aceleración en el tiempo de una estructura bajo condiciones de 
    vibración libre amortiguada. Ingresar con unidades consistentes.
    '''
    const = (xi*w*v0+xi**2*w**2*x0+x0*wD**2)/wD
    a = np.exp(-xi*w*t)*(-(xi*w*v0+const*wD)*np.cos(wD*t)+(const*xi*w-v0*wD)*np.sin(wD*t))
    return a

# %% Definición de variables
g = 9.81  # [m/s²]
M6 = 2000 # [kg/m²] carga losa
l = 6     # [m] lado losa
H = 20    # [m] altura de los pórticos
fpc = 210 # [kgf/cm²] f'c del concreto
SIGMA_MAX = 0.65*fpc*(100**2)*g # [N/m²]
TAU_MAX = 0.53*np.sqrt(fpc)*(100**2)*g # [N/m²]
# Condiciones iniciales
x0 = 0.009*H # [m] Posición inicial impacto natural, dirección x 0.9%
v0 = 8       # [m/s] Velocidad inicial, dirección x

# %% Cálculos
A = l*l   # [m²] Área losa
masa = M6*A  # [kg]
delta_max = 0.007*H # [m] Desplazamiento máximo asumido en dirección x
E = 15100*np.sqrt(fpc)*g*(100**2) # [N/m²] módulo de E. del Concreto NSR-10
s_a = 0.95*g # [m/s²] Aceleración asociada al desplazamiento máximo
fs = masa*s_a # [N] Fuerza que produce esta aceleración en la estructura

# %% Cálculo de rigidez
# Asumiendo el comportamiento elástico lineal del material, puede determinarse
# una rigidez mínima para las condiciones dadas del problema, bajo el enfoque de fuerza estática 
# equivalente (1.8.1, Chopra)
k_min = fs/delta_max # [N/m]
n_cols = 4 # Número de columnas
I_min = k_min*H**3/(12*n_cols*E) # [m^4]
b_min = (12*I_min)**(1/4) # [m] base mínima columna sección cuadrada
# Propiedades colocadas
# b = b_min
b = 0.85 # [m] base columna
h = 0.85 # [m] altura columna
I = b*h**3/12 # [m^4]
# Se toma esta rigidez dado que es la que aportan las columnas de forma
# perpendicular a su eje axial. Se considera una sección cuadrada para las
# columnas.
k_col = 12*E*I/(H**3) # [N/m]
k = n_cols*k_col # cuatro columnas
w = np.sqrt(k/masa) # [1/s] Frecuencia natural

# %% Vectores
delta_t = 0.01; # [s]
tt = np.arange(0,10, delta_t) # [s]
xxi = np.array([0.0, 0.05, 0.10, 0.30]) # [adim] -> c/cc
wD = np.sqrt(1-xxi**2)*w  # [1/s] Frecuencia amortiguada
cc = 2*xxi*masa*w # kg/s

# Definición de variables y arrays para almacenar los resultados y realizar los ciclos
m = len(xxi)
n = len(tt)
xx = np.empty((m,n))
vv = np.empty((m,n))
aa = np.empty((m,n))

# Desplazamientos, velocidades y aceleraciones
for i in range(m):
    xx[i] = x_t(tt, w, xxi[i], v0, x0, wD[i])
    vv[i] = v_t(tt, w, xxi[i], v0, x0, wD[i])
    aa[i] = a_t(tt, w, xxi[i], v0, x0, wD[i])

# Fuerzas de inercia, amortiguamiento y rigidez
f_i = masa*aa # N
f_a = (np.tile(cc,(len(tt),1)).T)*vv # N
f_r = k*xx # N

# Máximos
max_x = np.max(abs(xx), axis=1) # [m] Máximos desplazamientos
max_v = np.max(abs(vv), axis=1) # [m/s] Máximas velocidades
max_a = np.max(abs(aa), axis=1) # [m/s²] Máximas aceleraciones
max_f_i = np.max(abs(f_i), axis=1) # [N] Máximas fuerzas inerciales
# max_a*masa == max_f_i
max_f_r = np.max(abs(f_r), axis=1) # [N] Máximas fuerzas de rigidez - dinámicas
# F_r es equivalente, en esta estructura, a la fuerza cortante. Se reparte entre las columnas
V_col = max_f_r/n_cols
# Esfuerzos máximos
M_u = V_col*H # [N m] Momento último
sigma_max = M_u*(b/2)/I # [N/m²] Esfuerzo normal máximo en cada columna
tau_max = 3/2*V_col/(b*h) # [N/m²] Esfuerzo cortante máximo en cada columna
# %% Gráficas
plt.style.use("seaborn-paper")

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

for i in range(m):
    ax1.plot(tt,xx[i], label=r'$\xi=$'+ str(xxi[i]))
    ax2.plot(tt,vv[i], label=r'$\xi=$'+ str(xxi[i]))
    ax3.plot(tt,aa[i], label=r'$\xi=$'+ str(xxi[i]))

ax1.set_title("Desplazamientos")
ax1.set_ylabel(r'$x(t)\quad [m]$')
# ax1.set_xlabel(r'$t\quad [s]$')
ax1.legend(loc=0)
ax1.grid(True)
ax2.set_title("Velocidades")
ax2.set_ylabel(r'$v(t)\quad [m/s]$')
# ax2.set_xlabel(r'$t\quad [s]$')
ax2.legend(loc=0)
ax2.grid(True)
ax3.set_title("Aceleraciones")
ax3.set_ylabel(r'$a(t)\quad [m/s^2]$')
ax3.set_xlabel(r'$t\quad [s]$')
ax3.legend(loc=0)
ax3.grid(True)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

for i in range(m):
    ax1.plot(tt,f_r[i], label=r'$\xi=$'+ str(xxi[i]))
    ax2.plot(tt,f_a[i], label=r'$\xi=$'+ str(xxi[i]))
    ax3.plot(tt,f_i[i], label=r'$\xi=$'+ str(xxi[i]))

ax1.set_title("Fuerzas de rigidez")
ax1.set_ylabel(r'$F_r(t)\quad[N]$')
# ax1.set_xlabel(r'$t\quad [s]$')
ax1.legend(loc=0)
ax1.grid(True)
ax2.set_title("Fuerzas de amortiguamiento")
ax2.set_ylabel(r'$F_a(t)\quad[N]$')
# ax2.set_xlabel(r'$t\quad [s]$')
ax2.legend(loc=0)
ax2.grid(True)
ax3.set_title("Fuerzas de inercia")
ax3.set_ylabel(r'$F_i(t)\quad [N]$')
ax3.set_xlabel(r'$t\quad [s]$')
ax3.legend(loc=0)
ax3.grid(True)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

for i in range(m):
    ax1.plot(f_r[i],f_i[i], label=r'$\xi=$'+ str(xxi[i]))
    ax2.plot(f_a[i],f_i[i], label=r'$\xi=$'+ str(xxi[i]))
    ax3.plot(f_a[i],f_r[i], label=r'$\xi=$'+ str(xxi[i]))

ax1.set_title("Fuerzas de inercia vs Fuerzas de rigidez")
ax1.set_ylabel(r'$F_i(t)\quad [N]$')
ax1.set_xlabel(r'$F_r(t)\quad [N]$')
ax1.legend(loc=0)
ax1.grid(True)
ax1.axis('equal')
ax2.set_title("Fuerzas de inercia vs Fuerzas de amortiguamiento")
ax2.set_ylabel(r'$F_i(t)\quad [N]$')
ax2.set_xlabel(r'$F_a(t)\quad [N]$')
ax2.legend(loc=0)
ax2.grid(True)
ax2.axis('equal')
ax3.set_title("Fuerzas de rigidez vs Fuerzas de amortiguamiento")
ax3.set_ylabel(r'$F_r(t)\quad [N]$')
ax3.set_xlabel(r'$F_a(t)\quad [N]$')
ax3.legend(loc=0)
ax3.grid(True)
ax3.axis('equal')
plt.show()

fig = plt.figure()
for i in range(m):
    plt.plot(tt, f_i[i]+f_a[i]+f_r[i], label=r'$\xi=$'+ str(xxi[i]))
plt.legend(loc=0)
plt.title(r'$F_i+F_a+F_r=0$')
plt.ylabel("Fuerzas "r'$[N]$')
plt.xlabel(r'$t\quad [s]$')
plt.show()
# %%
