from cmath import cos as cos
from cmath import sin as sin
from math import pi
from pickletools import optimize
import string
from xml.etree.ElementTree import tostring
from matplotlib import colors
import scipy.optimize
from sympy import *
from sympy.core import symbol
from sympy.core.numbers import Pi
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

# объявление переменных, которые будут использоваться в расчетах
x, U0, U1, U2, U3, U4, U_dot_0, U_dot_1, U_dot_2, U_dot_3, U_dot_4, b, c = symbols('x U0 U1 U2 U3 U4 U_dot_0 U_dot_1 U_dot_2 U_dot_3 U_dot_4 b c')

# определение b и собственной функции #

b_ = 10
c_ = 1

def own_function(n,x) :
    return cos(n*x)

U_t = U0/2 + U1*own_function(1,x) + U2*own_function(2,x) + U3*own_function(3,x) + U4*own_function(4,x) 
U_dot = U_dot_0/2 + U_dot_1*own_function(1,x) + U_dot_2*own_function(2,x) + U_dot_3*own_function(3,x) + U_dot_4*own_function(4,x)
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
#zero_A = integrate(A, (x, 0, pi))

#sub_zero_A = zero_A.subs(U_dot_0, 0)
#coef_zero_A = (zero_A - sub_zero_A)/U_dot_0

zero_B = integrate(B, (x, 0, pi))

print(zero_B)
print(type(zero_B))
a = zero_B.to
print(a)

#zero_B = (zero_B - sub_zero_A)/coef_zero_A

print('{} = {}'.format(U_dot_0, zero_B))

#нахождение 2го интеграла
#first_A = integrate(A*own_function(1,x), (x, 0, pi))

#sub_first_A = first_A.subs(U_dot_1, 0)
#coef_first_A = (first_A - sub_first_A)/U_dot_1

first_B = integrate(B*own_function(1,x), (x, 0, pi))

#first_B = (first_B - sub_first_A)/coef_first_A

print('{} = {}'.format(U_dot_1, first_B))



#нахождение 3го интеграла
#second_A = integrate(A*own_function(2,x), (x, 0, pi))

#sub_second_A = second_A.subs(U_dot_2, 0)
#coef_second_A = (second_A - sub_second_A)/U_dot_2

second_B = integrate(B*own_function(2,x), (x, 0, pi))

#second_B = (second_B - sub_second_A)/coef_second_A

print('{} = {}'.format(U_dot_2, second_B))


#нахождение 4го интеграла
#third_A = integrate(A*own_function(3,x), (x, 0, pi))

#sub_third_A = third_A.subs(U_dot_3, 0)
#coef_third_A = (third_A - sub_third_A)/U_dot_3

third_B = integrate(B*own_function(3,x), (x, 0, pi))

#third_B = (third_B - sub_third_A)/coef_third_A

print('{} = {}'.format(U_dot_3, third_B))

#нахождение 5го интеграла
#third_A = integrate(A*own_function(3,x), (x, 0, pi))

#sub_third_A = third_A.subs(U_dot_3, 0)
#coef_third_A = (third_A - sub_third_A)/U_dot_3

fourth_B = integrate(B*own_function(4,x), (x, 0, pi))

#third_B = (third_B - sub_third_A)/coef_third_A

print('{} = {}'.format(U_dot_4, fourth_B))
###############################################################################################


try :
    print("start solve equation")
    sol = solve([first_B.subs(b, b_), second_B.subs(b, b_), third_B.subs(b, b_), fourth_B.subs(b, b_)], U1, U2, U3, U4, simplify=True, rational=False, particular=True)
    print("equation succesfully solved")
    print(sol)
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print('b={}:\n {}\n'.format(b_, soln))
except Exception as e :
    print(e)

legend = []

for s in soln :
    
    dot_stability = 1

    x11 = diff(first_B.subs(b,b_),U1)
    x11_ = complex(x11.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x12 = diff(first_B.subs(b,b_),U2)
    x12_ = complex(x12.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x13 = diff(first_B.subs(b,b_),U3)
    x13_ = complex(x13.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x14 = diff(first_B.subs(b,b_),U4)
    x14_ = complex(x14.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x21 = diff(second_B.subs(b,b_),U1)
    x21_ = complex(x21.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x22 = diff(second_B.subs(b,b_),U2)
    x22_ = complex(x22.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x23 = diff(second_B.subs(b,b_),U3)
    x23_ = complex(x23.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x24 = diff(second_B.subs(b,b_),U4)
    x24_ = complex(x24.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x31 = diff(third_B.subs(b,b_),U1)
    x31_ = complex(x31.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x32 = diff(third_B.subs(b,b_),U2)
    x32_ = complex(x32.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x33 = diff(third_B.subs(b,b_),U3)
    x33_ = complex(x33.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x34 = diff(third_B.subs(b,b_),U4)
    x34_ = complex(x34.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))
    
    x41 = diff(fourth_B.subs(b,b_),U1)
    x41_ = complex(x41.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x42 = diff(fourth_B.subs(b,b_),U2)
    x42_ = complex(x42.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x43 = diff(fourth_B.subs(b,b_),U3)
    x43_ = complex(x43.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))

    x44 = diff(fourth_B.subs(b,b_),U4)
    x44_ = complex(x44.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]}))
    print('-----------------------------------------------------------------------------------------------')

    yakobi_matrix = np.array([[x11_.real, x12_.real, x13_.real, x14_.real], [x21_.real, x22_.real, x23_.real, x24_.real], \
        [x31_.real, x32_.real, x33_.real, x34_.real], [x41_.real, x42_.real, x43_.real, x44_.real]], dtype=complex)

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
    
    t = 5  #переменная времени
     
    arr_y = []
    arr_x = []
    x_ = 0
    while x_ < pi :
        U_dot_0_ = zero_B.subs({U1:s[0], U2:s[1], U3:s[2], U4:s[3]})
        #интегрируя U0 с точкой по t получим:
        U0_ = U_dot_0_*t
        U_t_ = complex(U_t.subs({U0:U0_, U1:s[0], U2:s[1], U3:s[2], U4:s[3], x:x_}))
        arr_y.append(U_t_.real)
        arr_x.append(x_)
        x_ += 0.01

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
plt.show()