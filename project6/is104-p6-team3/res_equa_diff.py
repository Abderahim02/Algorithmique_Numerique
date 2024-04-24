import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from math import sqrt


##Implimentation de la méthode d'Euler
def step_euler(y, t, h, f):
    return y+h*f(t, y) #y{n+1}

##Implimentation de la méthode du point milieu
def step_mid_point(y, t, h, f):
    y_int=y+(h/2)*f(t,y) # un point intérmediere y_int = y_{n+(1/2)}
    return y+h*f(t+(h/2), y_int) #le résultat est y_{n+1}

##Implimentation de la méthode de Heun
def step_heun(y, t, h, f):
    p1=f(t,y)
    y2=y+h*p1
    p2=f(t+h, y2)
    return y+(h/2)*(p1+p2) #y{n+1}

##Implimentation de la méthode de Runge-Kutta d'ordre 4
def step_runge_kutta_4(y, t, h, f):
    p1=f(t,y)
    y2=y+(1/2)*h*p1
    p2=f(t+(h/2), y2)
    y3=y+(h/2)*p2
    p3=f(t+(2/h), y3)
    y4=y+h*p3
    p4=f(t+h, y4)
    return y+(h/6)*(p1+2*p2+2*p3+p4)

def norme_infinie(y):
    s=0
    for i in range(len(y)):
        s+=y[i]**2
    return sqrt(s)


##Cette fonction calcule le nombre N de pas de taille constante h
def meth_n_step(y0, t0, N, h, f, meth):
    t=[t0 for i in range(N)]
    y=[y0 for i in range(N)]
    for i in range(1, N):
        t[i]=t0+i*h
        y[i]=meth(y[i-1], t[i-1], h, f)
    return np.array(t), np.array(y)

def norm(y, z):
    L=[0 for i in range(len(y))]
    for i in range(len(y)):
        for j in range(len(z)):
            if(i==2*j):
                L[i]=L[i]-L[j]
    return np.linalg.norm(L)

# ##Cette fonction calcule une solution approché avec un paramètre d'erreur epsilon
def meth_epsilon(y0, t0, tf, eps, f, meth):
    N=1000
    h=(tf-t0)/N
    y_N = meth_n_step(y0, t0, N, h, f, meth)[1]
    y_2N = meth_n_step(y0, t0, 2*N, h/2, f, meth)[1]
    error = np.abs(norm(y_N, y_2N))
    while(error > eps):
            print(N)
            N=N*2
            h=(tf-t0)/N
            y_N=y_2N
            y_2N=meth_n_step(y0, t0, 2*N, h/2, f, meth)
            error = np.abs(norm(y_N, y_2N))
    return y_N

# ##Cette fonction calcule une solution approchée avec un paramètre d'erreur epsilon
# def meth_epsilon_m(y0, t0, tf, eps, F, meth):
#     N=1000
#     h=(tf-t0)/N
#     y_N = meth_n_step(y0, t0, N, h, F, meth)[1]
#     y_2N = meth_n_step(y0, t0, 2*N, h/2, F, meth)[1]
#     error = np.abs(norm(y_N, y_2N))
#     while(error > eps):
#             print(N)
#             N=N*2
#             h=(tf-t0)/N
#             y_N=y_2N
#             y_2N=meth_n_step(y0, t0, 2*N, h/2, F, meth)[1]
#             error = np.abs(norm(y_N, y_2N))
#     return y_N

# ##Cette fonction calcule n étapes d'une méthode numérique sur une équation différentielle avec une fonction F
# def meth_n_step_m(y0, t0, n, h, F, meth):
#     y=y0
#     t=t0
#     for i in range(n):
#         y=meth(y, t, h, F)
#         t=t+h
#     return t, y


##Cette fonction permet de dessiner le champ des tangentes de l'équation différentielle (on utilise quiver)


def plot_champ_tang_equ_diff(f, t0, tf, y0, yf, N):
    X, Y, U, V = [1 for i in range(0, N**2)], [1 for i in range(0, N**2)], [1 for i in range(0, N**2)], [1 for i in range(0, N**2)]
    hx, hy = (tf - t0)/N, (yf - y0)/N
    for i in range(0, N):
        for j in range(0, N):
            X[N*i + j] = t0 + i*hx
            Y[N*i + j] = y0 + j*hy
            V[N*i + j] = f(t0 + i*hx, y0 + j*hy)
    colors = np.arctan2(U, V)
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.viridis
    plt.quiver(X, Y, U, V, color=colormap(norm(colors)))
    plt.show()


def h(x,y):
    return x**2+y**2

#plot_champ_tang_equ_diff(h, -3, 3, -5, 5, 100)
############### Rkutta4 N dimension ############## Abderahim

def step_runge_kutta(y, t, h, f):
    k1 = f(y, t)
    k2 = f(y + (h/2)*k1, t + h/2)
    k3 = f(y + (h/2)*k2, t + h/2)
    k4 = f(y + h*k3, t + h)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
# Algorithme meth_n_step

def meth_n_step(y0, t0, N, h, f, step_method):
    y = np.zeros((N + 1, len(y0)))
    t = np.zeros(N + 1)
    y[0] = y0
    t[0] = t0

    for n in range(N):
        phi = step_method(y[n], t[n], h, f)
        y[n + 1] = phi
        t[n + 1] = t[n] + h
    return y, t
