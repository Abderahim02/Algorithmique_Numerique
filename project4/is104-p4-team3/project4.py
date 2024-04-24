
import numpy as np
import matplotlib.pyplot as plt
import cmath
import random

#---- newton-raphson-------
    # 1) H(U) * V = -f(U)

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


#solution for f(x)=0 with Newton-raphson with backtracking
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


#--------test: Newton_Raphson :with a simple quadratic function : f(x)=x^2 -4
def test_Raphson_Method():
    def f(x):
        return np.array([x[0]**2 - 4])
    def J(x):
        return np.array([[2 * x[0]]])
    U0 = np.array([1.0]) 
    N = 100               
    epsilon = 1e-6  

    result, convergence = Newton_Raphson(f, J, U0, N, epsilon)

    if convergence:
        print(f"Root found: {result[0]}")
    else:
        print("Newton-Raphson algorithm did not converge.")

#------------test:Newton-raphson with 2 dimensions -----------
    #        f1(x, y) = x^2 + y^2 - 4
    #        f2(x, y) = x * y - 1
def test_Raphson_Method_2d():
    def f(U):
        x, y = U
        return np.array([x**2 + y**2 - 4, x * y - 1])
    def J(U):
        x, y = U
        return np.array([[2 * x, 2 * y], [y, x]])

    U0 = np.array([1.0, 1.0])  # with this U0 , the algorithm does not converge
    U0 =np.array([2.0, 0.5])   # it converges with this choice 
    N = 100                    
    epsilon = 1e-6              
    result, convergence = Newton_Raphson(f, J, U0, N, epsilon)

    if convergence:
        print(f"Root found: x = {result[0]}, y = {result[1]}")
    else:
        print("Newton-Raphson algorithm did not converge.")

# returns a path list used to store the convergence trajectory
def Newton_Raphson_with_path(f, J, U0, N, epsilon, alpha=0.05 ):
    U =U0 
    path = [U]
    for i in range(N):
        F_U =f(U)
        J_U =J(U)
        if np.linalg.norm(F_U) < epsilon :
            return U , True, path

        V, _,_,_ = np.linalg.lstsq(J_U, -F_U, rcond=None)
        t = 1
        while np.linalg.norm(f(U + t * V))**2 > np.linalg.norm(F_U)**2:
            t = alpha * t 
        U = U + t*V
        path.append(U)
    return U , False, path 


# 3D function used to test the newton-raphson algorithm and draw the representation of the path of convergence
def plot_3d():    
    def f(x):
        return np.array([x[0]**2 + x[1]**2 - x[2],
                        x[0] + x[1] - x[2]**2,
                        np.sin(x[0]) + np.cos(x[1]) - x[2]])
    def J(x):
        return np.array([[2*x[0], 2*x[1], -1],
                        [1, 1, -2*x[2]],
                        [np.cos(x[0]), -np.sin(x[1]), -1]])
    U0 = np.array([1, 1, 1])
    N = 100
    epsilon = 1e-6
    result, convergence, path = Newton_Raphson_with_path(f, J, U0, N, epsilon)

    def draw_3D():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z1 = X**2 + Y**2
        Z2 = X + Y
        Z3 = np.sin(X) + np.cos(Y)

        ax.plot_surface(X, Y, Z1, alpha=0.3)
        ax.plot_surface(X, Y, Z2, alpha=0.3)
        ax.plot_surface(X, Y, Z3, alpha=0.3)

        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Représentation 3D de la convergence de la méthode de Newton-Raphson")
        plt.tight_layout()
        plt.show()
    draw_3D()

##############
def elastic_force(x, k, x0):
    return -k*(x-x0)

def elastic_force_derivative(x, k, x0):
    return -k

def centrifugal_force(X, k, X0):
    return np.array([[k*(X[0]-X0[0])], [k*(X[1]-X0[1])]])

def centrifu_force_jacobian(X, k, X0):
    
    return np.array([[k, 0], [0, k]])

def gravitational_force(X, k, X0):
    (x, y) = X
    (x0, y0) = X0
    return np.array([k*(x-x0)/(((x-x0)**2 + (y-y0)**2)**(3/2)), k*(y-y0)/(((x-x0)**2 + (y-y0)**2)**(3/2))])

def grav_force_jacobian(X, k, X0):
    (x, y) = X
    (x0, y0) = X0
    return np.array([[k/(((x-x0)**2 + (y-y0)**2)**(3/2)) - 3*k*(x-x0)**2/(((x-x0)**2 + (y-y0)**2)**(5/2)), -3*k*(x-x0)*(y-y0)/(((x-x0)**2 + (y-y0)**2)**(5/2))], [-3*k*(x-x0)*(y-y0)/(((x-x0)**2 + (y-y0)**2)**(5/2)), k/(((x-x0)**2 + (y-y0)**2)**(3/2)) - 3*k*(y-y0)**2/(((x-x0)**2 + (y-y0)**2)**(5/2))]])


###################### FIN PARTIE 2 A ########################

################### DEBUT PARTIE 2.b ##########################


#on calcule les racines de la fonction F(S,R)



# Find the roots of a polynomial using Bairstow's method.
# Returns:
#     roots (list): Complex roots of the polynomial.



def bairstow_b(a, r, s, g, roots):
    if g < 1:
        return None
    if g == 1 and a[1] != 0:
        roots.append(float(-a[0]) / float(a[1]))
        return None
    if g == 2:
        D = (a[1]**2.0) - 4.0 * a[2] * a[0]
        X1 = (-a[1] - cmath.sqrt(D)) / (2.0 * a[2])
        X2 = (-a[1] + cmath.sqrt(D)) / (2.0 * a[2])
        roots.append(X1)
        roots.append(X2)
        return None
    n = len(a)
    b = [0] * len(a)
    c = [0] * len(a)
    b[n - 1] = a[n - 1]
    b[n - 2] = a[n - 2] + r * b[n - 1]
    i = n - 3
    while i >= 0:
        b[i] = a[i] + r * b[i + 1] + s * b[i + 2]
        i -= 1
    c[n - 1] = b[n - 1]
    c[n - 2] = b[n - 2] + r * c[n - 1]
    i = n - 3
    while i >= 0:
        c[i] = b[i] + r * c[i + 1] + s * c[i + 2]
        i -= 1
    Din = (c[2] * c[2] - c[3] * c[1]) ** (-1.0)
    r = r + Din * (c[2] * (-b[1]) - c[3] * (-b[0]))
    s = s + Din * (-c[1] * (-b[1]) + c[2] * (-b[0]))
    if abs(b[0]) > 1E-4 or abs(b[1]) > 1E-4:
        return bairstow_b(a, r, s, g, roots)
    if g >= 3:
        Dis = (-r) ** 2.0 - 4.0 * 1.0 * (-s)
        X1 = (r - cmath.sqrt(Dis)) / 2.0
        X2 = (r + cmath.sqrt(Dis)) / 2.0
        roots.append(X1)
        roots.append(X2)
        return bairstow_b(b[2:], r, s, g - 2, roots)
	
# def Newton_Raphson_b(f, J, U0, N, epsilon):
#     U = U0 
#     for i in range(N):
#         F_U = f(U)
#         J_U = J(U)
#         if np.linalg.norm(F_U) < epsilon:
#             print(U)
#             return U, True
#         V = np.linalg.solve(J_U, -F_U)
#         U = U + V
#     return U, False
def Newton_Raphson_b(f, J, U0, N, epsilon):
    U = U0 # convertir U0 en float64
    # print(U)
    for i in range(N):
        F_U = f(U)
        J_U = J(U)
        if np.linalg.norm(F_U) < epsilon:
            # print(U)
            return U, True
        V = np.linalg.solve(J_U.astype(np.float64), -F_U.astype(np.float64)) # convertir J_U et -F_U en float64
        U = U + V.astype(np.float64) # convertir V en U.dtype
    return U, False

##this on ereturns only real roots
def bairstow_raphson(a, b, c, g, roots):
    if g < 1:
        return None
    if g == 1 and a[1] != 0:
        roots.append(float(-a[0]) / float(a[1]))
        return None
    if g == 2:
        D = (a[1]**2.0) - 4.0 * a[2] * a[0]
        X1 = (-a[1] - np.sqrt(D)) / (2.0 * a[2])
        X2 = (-a[1] + np.sqrt(D)) / (2.0 * a[2])
        roots.append(X1)
        roots.append(X2)
        return None
    quotient,reste = np.polydiv(a, [c, b, 1])
    R, S = reste[1], reste[0]
    # print("R,S", R, S)
    def F(x): #x=[b, c]
        if np.isnan(x[0]) :
            return np.zeros((2,1))
        return np.array([ R*x[0], S] , dtype = np.float64)
    quotient2,reste2 = np.polydiv(quotient, [c, b, 1])
    if(reste2.shape[0] <= 1):
        R1, S1 = 0, reste2[0]
    else :
        R1, S1 = reste2[1], reste2[0]
    # print("R1,S1", R1, S1)
    def jacobian_F(x): #x=[b, c]
        if np.isnan(x[0])  or np.isnan(x[1]):
            return np.zeros((2,2))
        return np.array([
            [x[0]*R1 - S1, x[1]*R1 ],
            [-R1, -S1]
        ], dtype = np.float64)
    d = b**2 - 4*c
    if d >=0:
        r1 = (-b + np.sqrt(b**2 - 4*c)) / 2
        r2 = (-b - np.sqrt(b**2 - 4*c)) / 2
    else:
        r1 = (-b + np.sqrt(-d)) / 2
        r2 = (-b - np.sqrt(-d)) / 2
    #x0 = np.array([R, S])
    #x0 = np.array([0.01,0.05 ])
    x0 = np.array([r1, r2])
    x, bol = Newton_Raphson_BT(F, jacobian_F, x0, 10, 10e-3,0.5 )
    #print(x,bol)
    b0 , c0 = x[0], x[1] 
    # Check for convergence
    b += b0
    c += c0
    # if abs(b0) > 1E-4 or abs(c0) > 1E-4*abs(c0) :
    #     return bairstow_raphson(a, b, c, g, roots)
     
    if abs(b0) < 1E-4 or abs(c0) < 1E-4*abs(c0) :
        return None
    # If we have converged, solve for the roots
    if g >= 3:
        Dis = (b) ** 2.0 - 4.0 * 1.0 * (c)
        if Dis >= 0:
            X1 = (b - np.sqrt(Dis)) / 2.0
            X2 = (b + np.sqrt(Dis)) / 2.0
            roots.append(X1)
            roots.append(X2)
        else:
            # X1 = complex(b0/2 , np.sqrt(c0-(b0/2)**2))
            # X2 = complex(b0/2 ,  - np.sqrt(c0-(b0/2)**2))
            X1 = complex(b/2 , np.sqrt(-Dis)/2)
            X2 = complex(b/2 , -np.sqrt(-Dis)/2)
            roots.append(X1)
            roots.append(X2)
        # X1 = complex(b0/2 , np.sqrt(-(c0-(b0/2)**2)))
        # X2 = complex(b0/2 ,  - np.sqrt(-(c0-(b0/2)**2)))
        # roots.append(X1)
        # roots.append(X2)
        return bairstow_raphson(a[2:], b, c, g - 2, roots)

def test_bairstow():
    # '''     We will find all the roots of
    #    -12 + 2x + 3x^2 - 6x^3 + 6x^4 - x^5
    # '''
    g = 7
    roots = []
    a = [-12, 2, 3, -6, 6, -1, 5, -2]
    a_rev=[-2, 5,-1, 6, -6, 3, 2, -12]
    b = 0.1
    c = 0.2
    bairstow_b(a,b,c,g,roots)
    ##results 
    z_th = np.roots(a_rev)
    z_exp = roots
    print(z_th)
    print(z_exp)
    #theric reuluts
    x_th = z_th.real
    y_th = z_th.imag
    ##experimental results
    roots[len(roots) - 1 ] = roots[len(roots) - 1 ] + 0j
    x_exp = [x.real + 0.1*x.real if isinstance(x, complex) else x for x in roots]
    y_exp = [x.imag +0.1*x.imag if isinstance(x, complex) else x for x in roots]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_th, y_th, 'x', label="résultats théoriques")
    ax.plot(x_exp, y_exp, 'o', label="résultats expérimentaux")
    ax.set_xlim([min(x_th) - 1, max(x_th) + 1])
    ax.set_ylim([min(y_th) - 1, max(y_th) + 1])
    plt.legend()
    ax.set_xlabel('Réel')
    ax.set_ylabel('Imaginaire')
    plt.show()

###################### DEBUT PARTIE 2.C ###############################

def g(U):
    x, y, z, t = U 
    return np.array([(2*x)/(x**2 - 1), (2*y)/(y**2 - 1), (2*z)/(z**2 - 1), (2*t)/(t**2 - 1)])

def Jg(U):
    x, y, z, t = U 
    return np.array([[(-x**2 - x**2 - 2)/(x**4 - x**2 - x**2 + 1)], [(-y**2 - y**2 - 2)/(y**4 - y**2 - y**2 + 1)], [(-z**2 -z**2 - 2)/(z**4 - z**2 - z**2 + 1)], [(-t**2 - t**2 - 2)/(t**4 - t**2 - t**2 + 1)]]) 

def test():
    U1 = np.array([0.6, 0.6, 0.6, 0.6])
    M = 100
    eps = 1e-6
    sol, res = Newton_Raphson(g, Jg, U1, M, eps)
    print("\n")
    if res: 
        print(f"Results found : x1 = {sol[0]}, x2 = {sol[1]}, x3 = {sol[2]}, x4 = {sol[3]}")
    else:
        print("L'exemple choisi ne converge pas")
test()
###################### DEBUT PARTIE 2.C ###############################


def main():
    test_Raphson_Method()
    test_bairstow()
    test_Raphson_Method_2d()
    plot_3d()
main()

