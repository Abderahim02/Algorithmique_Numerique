from matplotlib import cm
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#solution for f(x)=0 with Newton-raphson
def Newton_Raphson(f, J, U0, N, epsilon):
    U =U0 
    for i in range(N):
        F_U =f(U)
        J_U =J(U)
        if np.linalg.norm(F_U) < epsilon :
            return U , True
        V, _,_,_ = np.linalg.lstsq(J_U, -F_U, rcond=None)
        U= U + V
    return U , False

def Newton_Raphson_BT(f, J, U0, N, epsilon, alpha=0.05 ):
    U =U0 
    for i in range(N):
        F_U =f(U)
        J_U =J(U)
        if np.linalg.norm(F_U) < epsilon :
            return U , True

        V, _,_,_ = np.linalg.lstsq(J_U, -F_U, rcond=None)
        
        t = 1
        while np.linalg.norm(f(U + t * V))**2 > np.linalg.norm(F_U)**2:
            t = alpha * t 
        U = U + t*V
    return U , False

#V a vector with N elements
def E_function(vecteur1,vecteur2):
    vecteur=[vecteur1[0],vecteur2[0]]
    somme = 0
    for i in range(0, len(vecteur1)):
        vecteur=[vecteur1[i],vecteur2[i]]
        print(vecteur)
        somme += math.log(np.abs(vecteur[i] + 1)) + math.log(np.abs(vecteur[i]-1))
        for j in range(len(vecteur1)):
            if j != i:
                somme += 0.5 * math.log(np.abs(vecteur[i] - vecteur[j]))
    return somme


def E_2d(X, Y):
    if (X != Y):
        return math.log(np.abs(X+1)) + math.log(np.abs(X-1)) + 0.5*math.log(np.abs(X-Y)) + math.log(np.abs(Y+1)) + math.log(np.abs(Y-1)) + 0.5*math.log(np.abs(Y-X))
    return math.log(np.abs(X+1)) + math.log(np.abs(X-1)) + math.log(np.abs(Y+1)) + math.log(np.abs(Y-1))

def Nabla_E_X(vecteur):
    n = len(vecteur)
    V = np.zeros(n) 
    for i in range (0, n) :
        V[i]=1/(vecteur[i]+1)+1/(vecteur[i]-1)
        for j in range(n):
            if j!=i:
                V[i]=V[i]+1/2*(vecteur[i]-vecteur[j])
    return V




# Définir les listes X et Y

def J_Nabla_E(vecteur):
    n=len(vecteur)
    J = np.zeros((n,n))
    for i in range (0, n):
        for j in range (0, n):
            if (i == j):
                somme =0
                for k in range (0,n):
                    if (k!=i):
                        if (vecteur[i] - vecteur[k]) != 0:
                            somme += -1/(vecteur[i]-vecteur[k])**2
                            J[i][j] = -1/(vecteur[i]+1)*2 - 1/(vecteur[i]-1)*2 + 0.5*somme
            else :
                if (vecteur[i] - vecteur[j] )!= 0:
                    J[i][j] = 0.5* (1/(vecteur[i]-vecteur[j])**2)
    return J

def draw_points():
    X = np.linspace(-0.99, 0.99, 100)
    Y = np.linspace(-0.99, 0.99, 100)
    finale = Newton_Raphson_BT(Nabla_E_X, J_Nabla_E, X,10, 10e-7, 0.5)[0]

    # Créer la grille de points en 3D
    x, y = np.meshgrid(X, Y)

    E_grid = np.zeros((len(X),len(Y)))
    for i in range (len(x)):
        for j in range (len(y[0])):
            E_grid[i][j] = E_2d(x[i][j],y[i][0])
    # Créer la figure en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer la surface
    ax.plot_surface(x, y, E_grid)

    # Ajouter des étiquettes aux axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Afficher la figure
    plt.show()
import numpy as np
from numpy.polynomial import legendre
import matplotlib.pyplot as plt

def draw_legendre():
    X = np.linspace(-0.99, 0.99, 10)
    Y = np.linspace(-0.99, 0.99, 10)
    finale = Newton_Raphson_BT(Nabla_E_X, J_Nabla_E, X, 10, 10e-7, 0.5)[0]
    # Créer une figure et un ensemble d'axes
    fig, ax = plt.subplots()
    # Tracer les points du tableau final
    ax.plot( finale, np.zeros(len(finale)), 'x',color='r')
    # Boucle sur les degrés et trace les courbes des polynômes de Legendre
    degrees = np.arange(2, 7)
    for n in degrees:
        p = legendre.Legendre.basis(n)
        dp = p.deriv()
        x = np.linspace(-1, 1, 1000)
        y = dp(x)
        ax.plot(x, y, label=f"n={n}")
    # Ajouter une légende et des étiquettes d'axe
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Afficher la figure
    plt.show()
def main():
    draw_points()
    draw_legendre()
main()