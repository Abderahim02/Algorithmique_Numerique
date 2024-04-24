


###############Abderahim################""


from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from res_equa_diff import *

mA = 6.0
mB = 0.01
G = 1
xA = 1.0
yA = 0

def F(Y, t):
    # Y is the vector of state [xB, yB, dxB/dt, dyB/dt]
    xB, yB, dxB, dyB = Y
    r = ((xB - xA)**2 + (yB - yA)**2)**0.5
    ddxB = G * mA * (xA - xB) / r**3
    ddyB = G * mA * (yA - yB) / r**3
    return np.array([dxB, dyB, ddxB, ddyB])

def plot_mouvement_B():
    xB0 = 2.5  # position initiale de B en mètres
    yB0 = 0
    vB = 2  # vitesse de B en mètres/seconde
    t0 = 0  # temps initial en secondes
    tf = 60 # temps final en secondes
    N = 100000  # nombre de pas
    h = (tf - t0) / N  # pas de temps

    # Définir l'état initial
    y0 = [xB0, yB0, 0, vB]

    # Appliquer la méthode de Runge-Kutta
    y, t = meth_n_step(y0, t0, N, h, F, step_runge_kutta)
    # Trajectoire de B dans le plan (xB, yB)
    xB = y[:, 0]
    yB = y[:, 1]

    # Afficher la trajectoire de B
    fig, ax = plt.subplots()
    ax.plot(xA, yA, 'ro', label='A')
    ax.plot(xB, yB)
    ax.set_xlabel('xB')
    ax.set_ylabel('yB')
    ax.set_aspect('equal')
    fig.set_size_inches(8, 6)  # Adjust the figure size

    plt.show()


def F_3corps(y, t):
    xC, yC, vxC, vyC = y
    # la distance entre A et C
    rC_A = np.sqrt(xC**2 + yC**2)

    # la distance entre A et C
    rC_B = np.sqrt((xC - 1)**2 + yC**2)
    rC_A3 = rC_A**3
    rC_B3 = rC_B**3

    # remplacer les valeurs des derivee dans le systeme
    dxC_dt = vxC
    dyC_dt = vyC
    dvxC_dt = -G * mA * xC / rC_A3 - G * mB * (xC - 1) / rC_B3
    dvyC_dt = -G * mA * yC / rC_A3 - G * mB * yC / rC_B3
    
    # dy_dt = [dxC_dt, dyC_dt, dvxC_dt, dvyC_dt]
    return np.array([dxC_dt, dyC_dt, dvxC_dt, dvyC_dt])

def solution_2():
    xC0 = 5.0
    yC0 = 2.0 
    dxC0 = 0.0 
    dyC0 = 1.0 
    Y0 = np.array([xC0, yC0, dxC0, dyC0])
    # t = np.linspace(0, 60, 100000) # time array
    # sol = odeint(F_3corps, Y0, t)

    t0 = 0  # temps initial en secondes
    tf = 60 # temps final en secondes
    N = 100000  # nombre de pas
    h = (tf - t0) / N  # pas de temps

    sol, t = meth_n_step(Y0, t0, N, h, F_3corps, step_runge_kutta)
    # Extract the xB and yB values from the solution array
    rB = 1.0
    xB = xA + np.cos(2*np.pi*t/(2*np.pi))
    yB = yA + np.sin(2*np.pi*t/(2*np.pi))
    xC = sol[:, 0]
    yC = sol[:, 1]
    fig, ax = plt.subplots()
    ax.plot(xC, yC, label = 'C')
    ax.plot(xB, yB, 'g', label='B')
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectoire du système à trois corps')
    ax.set_aspect('equal')
    plt.show()

def nouveau_ref():
    xC0 = 5.0 
    yC0 = 2.0 
    dxC0 = 0.0 
    dyC0 = 2.0 
    Y0 = np.array([xC0, yC0, dxC0, dyC0])
    t = np.linspace(0, 25, 1000) 
    # Calcul de la trajectoire de C et de B (comme précédemment)
    sol_C = odeint(F_3corps, Y0, t)
    xB = np.cos(2*np.pi*t/(2*np.pi))
    yB = np.sin(2*np.pi*t/(2*np.pi))
    t0=20
    # Transformation de la trajectoire de C
    theta = -2 * np.pi * t0 / (2 * np.pi)  # Angle de rotation
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])  # Matrice de rotation
    C_rotated = R @ sol_C[:, :2].T  # Transformation des coordonnées de C
    xC_rotated, yC_rotated = C_rotated[0], C_rotated[1]  # Récupération des coordonnées

    # Tracé de la trajectoire de C dans le nouveau référentiel
    fig, ax = plt.subplots()
    ax.plot(xC_rotated, yC_rotated)
    ax.plot(xB, yB, 'g', label='B')
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectoire de C dans le référentiel fixe de A et B')
    ax.set_aspect('equal')
    plt.show()
def nouveau_ref_2():
    xC0 = 5.0 
    yC0 = 2.0 
    dxC0 = 0.0 
    dyC0 = 2.0 
    Y0 = np.array([xC0, yC0, dxC0, dyC0])
    t = np.linspace(0, 25, 1000) 
    # Calcul de la trajectoire de C et de B (comme précédemment)
    sol = odeint(F_3corps, Y0, t)
    # Obtenir les coordonnées de la trajectoire de C
    xC = sol[:, 0]
    yC = sol[:, 1]

    # Obtenir les coordonnées de la trajectoire de B
    xB = np.cos(2*np.pi*t)
    yB = np.sin(2*np.pi*t)

    # Transformer les coordonnées de la trajectoire de C en appliquant la rotation
    C_rotated = np.zeros_like([xC, yC])
    for i in range(len(t)):
        R = np.array([[np.cos(2*np.pi*t[i]), -np.sin(2*np.pi*t[i])],
                    [np.sin(2*np.pi*t[i]), np.cos(2*np.pi*t[i])]])
        C_rotated[:, i] = R @ np.array([xC[i] - xB[i], yC[i] - yB[i]])

    # Tracer la trajectoire résultante
    fig, ax = plt.subplots()
    ax.plot(C_rotated[0], C_rotated[1], label='C')
    ax.plot(xB, yB, 'g', label='B')
    ax.plot(xA, yA, 'ro', label='A')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectoire du système à trois corps')
    plt.show()

if __name__ == "__main__":
    plot_mouvement_B()
    solution_2()
    nouveau_ref()
    nouveau_ref_2()








