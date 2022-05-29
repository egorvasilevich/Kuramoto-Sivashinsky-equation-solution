from cmath import cos as cos
from cmath import sin as sin
from math import pi
from sympy import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# объявление переменных, которые будут использоваться в расчетах
x, U0, U1, U2, U3, U4, U_dot_0, U_dot_1, U_dot_2, U_dot_3, b, c = symbols('x U0 U1 U2 U3 U4 U_dot_0 U_dot_1 U_dot_2 U_dot_3 b c')

# определение b и собственной функции #

b_ = 2
c_ = 1
t_ = [0,1,5]  #переменная времени

def own_function(n,x) :
    return cos(n*x)

U_t = U0/2 + U1*own_function(1,x) + U2*own_function(2,x) + U3*own_function(3,x)
U_dot = U_dot_0/2 + U_dot_1*own_function(1,x) + U_dot_2*own_function(2,x) + U_dot_3*own_function(3,x)
#########################################


Ux = diff(U_t, x)
Uxx = diff(Ux, x)
Uxxx = diff(Uxx, x)
Uxxxx = diff(Uxxx, x)

######################################
A = U_dot
B = - Uxxxx - (b*Uxx) - c*(Ux*Ux)
######################################

A = (2/pi)*A.subs({c: c_})
B = (2/pi)*B.subs({c: c_})


#нахождение интегралов
################################################################################
#нахождение 1го интеграла
zero_B = integrate(B, (x, 0, pi))
print('{} = {}'.format(U_dot_0, zero_B))

#нахождение 2го интеграла
first_B = integrate(B*own_function(1,x), (x, 0, pi))
print('{} = {}'.format(U_dot_1, first_B))

#нахождение 3го интеграла
second_B = integrate(B*own_function(2,x), (x, 0, pi))
print('{} = {}'.format(U_dot_2, second_B))

#нахождение 4го интеграла
third_B = integrate(B*own_function(3,x), (x, 0, pi))
print('{} = {}'.format(U_dot_3, third_B))

###############################################################################################

try :
    print("start solve equation")
    sol = solve([first_B.subs(b, b_), second_B.subs(b, b_), third_B.subs(b, b_)], U1, U2, U3)
    print("equation succesfully solved")
    print(sol)
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print('b={}:\n {}\n'.format(b_, soln))
except Exception as e :
    print(e)

###############################################################################################
###############################################################################################
###############################################################################################
#|
#V

t_array = [] # 10 times by 40 values 0 .. 9
x_array = [] #  10 times by 40 values 0 .. pi
u_t_array = [] # multiarray || 10 times by 40 values of u_t

my_colors = [
    'b',#	blue
    'g',#	green
    'r',#	red
    'c',#	cyan
    'm',#	magenta
    'y',#	yellow
    'k',#	black
    'w' #	white
]

for count_solutions in range(len(soln)) :
    u_t_array.append([])

print(u_t_array)

t_range = 100

legend = []
for t in range(t_range) :
    for s in soln :

        dot_stability = 1

        x11 = diff(first_B.subs(b,b_),U1)
        x11_ = complex(x11.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x12 = diff(first_B.subs(b,b_),U2)
        x12_ = complex(x12.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x13 = diff(first_B.subs(b,b_),U3)
        x13_ = complex(x13.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x21 = diff(second_B.subs(b,b_),U1)
        x21_ = complex(x21.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x22 = diff(second_B.subs(b,b_),U2)
        x22_ = complex(x22.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x23 = diff(second_B.subs(b,b_),U3)
        x23_ = complex(x23.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x31 = diff(third_B.subs(b,b_),U1)
        x31_ = complex(x31.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x32 = diff(third_B.subs(b,b_),U2)
        x32_ = complex(x32.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x33 = diff(third_B.subs(b,b_),U3)
        x33_ = complex(x33.subs({U1:s[0], U2:s[1], U3:s[2]}))

        print('-----------------------------------------------------------------------------------------------')

        yakobi_matrix = np.array([[x11_.real, x12_.real, x13_.real], [x21_.real, x22_.real, x23_.real], \
            [x31_.real, x32_.real, x33_.real]], dtype=complex)

        #находим собственные значения и вектора
        wa, va = LA.eig(yakobi_matrix)

        check_lower = 0
        check_upper = 0
        check_zero = 0
        for k in wa :
            if round(k.real,8) == 0 :
                check_zero += 1
            if k.real < 0 :
                check_lower += 1
            if k.real > 0 :
                check_upper += 1
        if check_zero != 0 :
            dot_stability = 0
            print("Точка V{} {} явялется точкой смены устойчивости при b = {}, wa = {}".format(soln.index(s)+1, s, b_, wa))
        elif check_upper != 0 :
            dot_stability = -1
            print('точка V{} {} равновесия не устойчива'.format(soln.index(s)+1, s))
        elif check_lower == 3 :
            print('точка V{} {} асимптотически устойчива'.format(soln.index(s)+1, s))

        arr_y = []
        arr_x = []
        x__ = 0
        while x__ < pi :
            #U_t_t_ = find_u_t(zero_B_=zero_B, U_t_=U_t, sol1=s[0], sol2=s[1], sol3=s[2], x_=x__, t_=t)
            U_dot_0_ = zero_B.subs({U1:s[0], U2:s[1], U3:s[2]})
            #интегрируя U0 с точкой по t получим:
            U0_ = U_dot_0_*t
            U_t_ = complex(U_t.subs({U0:U0_, U1:s[0], U2:s[1], U3:s[2], x:x__}))
            
            if soln.index(s) == 0 :
                t_array.append(t) # 40 times append t
                x_array.append(x__) # 40 times append 
            
            u_t_array[soln.index(s)].append(U_t_.real)

            x__ += 0.08

        stability = ''
        if dot_stability == 0 :
            stability = 'смена устойчивости'
        elif dot_stability == -1 :
            stability = 'неустойчива'
        else :
            stability = 'устойчива'

        print('-----------------------------------------------------------------------------------------------')

print(f"{t_array} \n ############# \n {x_array} \n ########### \n {u_t_array} \n")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(u_t_array)) :
    x_plot = np.reshape(t_array, (t_range, 40))
    y_plot = np.reshape(x_array, (t_range, 40))
    z_plot = np.reshape(u_t_array[i], (t_range, 40))

    ax.plot_surface(x_plot, y_plot, z_plot, color=my_colors[i%8])

ax.set_xlabel('T Label')
ax.set_ylabel('X Label')
ax.set_zlabel('U_t Label')

plt.show(block=False)
#A
#|
###############################################################################################
###############################################################################################
###############################################################################################


legend = []
for t in t_ :
    plt.figure()

    for s in soln :

        dot_stability = 1

        x11 = diff(first_B.subs(b,b_),U1)
        x11_ = complex(x11.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x12 = diff(first_B.subs(b,b_),U2)
        x12_ = complex(x12.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x13 = diff(first_B.subs(b,b_),U3)
        x13_ = complex(x13.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x21 = diff(second_B.subs(b,b_),U1)
        x21_ = complex(x21.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x22 = diff(second_B.subs(b,b_),U2)
        x22_ = complex(x22.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x23 = diff(second_B.subs(b,b_),U3)
        x23_ = complex(x23.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x31 = diff(third_B.subs(b,b_),U1)
        x31_ = complex(x31.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x32 = diff(third_B.subs(b,b_),U2)
        x32_ = complex(x32.subs({U1:s[0], U2:s[1], U3:s[2]}))

        x33 = diff(third_B.subs(b,b_),U3)
        x33_ = complex(x33.subs({U1:s[0], U2:s[1], U3:s[2]}))

        print('-----------------------------------------------------------------------------------------------')

        yakobi_matrix = np.array([[x11_.real, x12_.real, x13_.real], [x21_.real, x22_.real, x23_.real], \
            [x31_.real, x32_.real, x33_.real]], dtype=complex)

        #находим собственные значения и вектора
        wa, va = LA.eig(yakobi_matrix)

        check_lower = 0
        check_upper = 0
        check_zero = 0
        for k in wa :
            if round(k.real,8) == 0 :
                check_zero += 1
            if k.real < 0 :
                check_lower += 1
            if k.real > 0 :
                check_upper += 1
        if check_zero != 0 :
            dot_stability = 0
            print("Точка V{} {} явялется точкой смены устойчивости при b = {}, wa = {}".format(soln.index(s)+1, s, b_, wa))
        elif check_upper != 0 :
            dot_stability = -1
            print('точка V{} {} равновесия не устойчива'.format(soln.index(s)+1, s))
        elif check_lower == 3 :
            print('точка V{} {} асимптотически устойчива'.format(soln.index(s)+1, s))

        arr_y = []
        arr_x = []
        x__ = 0
        while x__ < pi :
            #U_t_t_ = find_u_t(zero_B_=zero_B, U_t_=U_t, sol1=s[0], sol2=s[1], sol3=s[2], x_=x__, t_=t)
            U_dot_0_ = zero_B.subs({U1:s[0], U2:s[1], U3:s[2]})
            #интегрируя U0 с точкой по t получим:
            U0_ = U_dot_0_*t
            U_t_ = complex(U_t.subs({U0:U0_, U1:s[0], U2:s[1], U3:s[2], x:x__}))
            arr_y.append(U_t_.real)
            arr_x.append(x__)
            x__ += 0.08

        plt.plot(arr_x, arr_y)

        stability = ''
        if dot_stability == 0 :
            stability = 'смена устойчивости'
        elif dot_stability == -1 :
            stability = 'неустойчива'
        else :
            stability = 'устойчива'

        legend.append('V{}- {} {} t = {}'.format(soln.index(s)+1, stability, s, t))
        print('-----------------------------------------------------------------------------------------------')
    plt.legend(legend, loc='upper right')
    plt.show(block=False)
    legend.clear()
plt.show()