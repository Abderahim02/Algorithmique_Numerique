from res_equa_diff import *

t0 = -5
tf = 5
y0 = 1
eps=0.00001

def fonction_1(t, y):
    return y/(1+t**2)

def sol_exact_1(y0, t0, t):
    K=3#y0*np.exp(np.arctan(t0))
    return K*np.exp(np.arctan(t))

def plot_test_sol_1(y0, t0, tf, eps, f, sol):
    y_N_euler=meth_epsilon(y0, t0, tf, eps, f, step_euler)
    X_1=np.linspace(t0, tf, len(y_N_euler))
    y_N_heun=meth_epsilon(y0, t0, tf, eps, f, step_heun)
    X_2=np.linspace(t0, tf, len(y_N_heun))
    y_N_midp=meth_epsilon(y0, t0, tf, eps, f, step_mid_point)
    X_3=np.linspace(t0, tf, len(y_N_midp))
    y_N_rk4=meth_epsilon(y0, t0, tf, eps, f, step_runge_kutta_4)
    X_4=np.linspace(t0, tf, len(y_N_rk4))
    sol_exact=np.array([sol(y0, t0, x) for x in X_4])

    plt.plot(X_1, y_N_euler, label="méthode d'euler")
    plt.plot(X_2, y_N_heun, label="méthode d'heun")
    plt.plot(X_3, y_N_midp, label="méthode de mid point")
    plt.plot(X_4, y_N_rk4, label="méthode de rk4")
    plt.plot(X_4, sol_exact, label="la solution exacte")
    plt.title("La solution par différentes méthodes")
    plt.legend()
    plt.show()

def plot_test_sol_2(y0, t0, tf, eps, f, sol):
    y_N_euler=meth_epsilon(y0, t0, tf, eps, f, step_euler)
    X_1=np.linspace(t0, tf, len(y_N_euler))
    y_N_heun=meth_epsilon(y0, t0, tf, eps, f, step_heun)
    X_2=np.linspace(t0, tf, len(y_N_heun))
    y_N_midp=meth_epsilon(y0, t0, tf, eps, f, step_mid_point)
    X_3=np.linspace(t0, tf, len(y_N_midp))
    y_N_rk4=meth_epsilon(y0, t0, tf, eps, f, step_runge_kutta_4)
    X_4=np.linspace(t0, tf, len(y_N_rk4))
    sol_exact=np.array([sol(y0, t0, x) for x in X_4])

    plt.plot(X_1, y_N_euler[:, 0], label="méthode d'euler")
    plt.plot(X_2, y_N_heun[:, 0], label="méthode d'heun")
    plt.plot(X_3, y_N_midp[:, 0], label="méthode de mid point")
    plt.plot(X_4, y_N_rk4[:, 0], label="méthode de rk4")
    plt.plot(X_4, sol_exact[:, 0, 0], label="la solution exacte")
    plt.title("La solution par différentes méthodes")
    plt.legend()
    plt.show()

plot_test_sol_1(y0, t0, tf, eps, fonction_1, sol_exact_1)

y0 = np.array([[1], [0]])
def fonction_2(t, y):
    return np.array([-y[1],y[0]])

def sol_exact_2(y0, t0, t):
    K=1#np.array([y0[0]*[np.cos(t0)], y0[1]*[np.sin(t0)]])
    return K*np.array([[np.cos(t)], [np.sin(t)]])

plot_test_sol_2(y0, t0, tf, eps, fonction_2, sol_exact_2)


# Création du graphe de convergence en fonction des différentes méthodes
# t0 = -5
# tf = 5
# y0 = 1
# K = y0/(np.exp(np.arctan(t0)))
# lim = 8


# def f(t, y):
#     return y/(1+t**2)

# def y(t):
#     return K*np.exp(np.arctan(t))

# n = [10**i for i in range(1,lim)]

# def array_values(y0, t0, tf,  f, meth, lim):
#     array = np.zeros([1, lim - 1])[0]
#     for i in range(1, lim) :
#         t, y_p = meth_n_step(y0, t0, 10**i, (tf-t0)/10**i, f, meth)
#         val = np.array([y(h) for h in t])
#         array[i - 1] = max(val-y_p)
#     return array
    


# euler = array_values(y0, t0, tf, f, step_euler, lim)
# mid_point = array_values(y0, t0, tf, f, step_mid_point, lim)
# heun = array_values(y0, t0, tf,f,  step_heun, lim)
# runge_k = array_values(y0, t0, tf, f, step_runge_kutta_4, lim)

# plt.grid()


# plt.plot(n, euler, label='Euler') 
# plt.plot(n, mid_point, label='Middle Point')
# plt.plot(n, heun, label='Heun')
# plt.plot(n, runge_k, label='Runge Kutta')

# plt.yscale("log")
# plt.xscale("log")

# plt.xlabel("Nombre de subdivisions")
# plt.ylabel("Précision du résultat")
# plt.title("Précision du calcul de l'équation différentielle y'= y/(1+t**2) en fonction du nombre de subdivisions, pour chaque méthode de résolution")

# plt.legend()

# plt.show()