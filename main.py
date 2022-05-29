from cmath import cos as cos
from cmath import sin as sin
from math import pi
from sympy import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# объявление переменных, которые будут использоваться в расчетах
x, U0, U1, U2, U3, U4, U_dot_0, U_dot_1, U_dot_2, U_dot_3, b, c = symbols('x U0 U1 U2 U3 U4 U_dot_0 U_dot_1 U_dot_2 U_dot_3 b c')

# определение b, c и собственной функции
b_ = 8 # с _ т.к. эти символы используются в уравнении
c_ = 1 # -||-

def own_function(n,x) :
    return cos(n*x)

# определение изначальной задачи Ut
U_t = U0/2 + U1*own_function(1,x) + U2*own_function(2,x) + U3*own_function(3,x)
U_dot = U_dot_0/2 + U_dot_1*own_function(1,x) + U_dot_2*own_function(2,x) + U_dot_3*own_function(3,x)

# дифференцируем U по x 4 раза
Ux = diff(U_t, x)
Uxx = diff(Ux, x)
Uxxx = diff(Uxx, x)
Uxxxx = diff(Uxxx, x)

# разделяем уравнение на две части А и В
A = U_dot
B = - Uxxxx - (b*Uxx) - c*(Ux*Ux)

# заранее подставляем коэфициент 2/pi и в уравнении заменяем c на заданное значение 
A = (2/pi)*A.subs({c: c_})
B = (2/pi)*B.subs({c: c_})

# далее считаем интегралы для четырех уравнений
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

# далее решаем полученную систему уравнений
try :
    sol = solve([first_B.subs(b, b_), second_B.subs(b, b_), third_B.subs(b, b_)], U1, U2, U3)
    print(sol) # все найденные решения системы
    soln = [tuple(v.evalf() for v in s) for s in sol]
    print('b={}:\n {}\n'.format(b_, soln))
except Exception as e :
    print(e)

# определяем необходимые структуры для сохранения решений
t_array = [] # массив содержащий значения t ( 6 раз по 40 значений 0 <= t <= 6)
x_array = [] # массив содержащий значения x ( 6 раз по 40 значений 0 <= x <= pi)
u_t_array = [] # многомерный массив, который содержит массивы решений для Ut. Размерность массива = кол-ву решений. Каждый массив состоит из 6 подмассивов по 40 значений Ut
legend = [] # массив содержащий информацию об устойчивости решения, собственно решения и его индекса. Используется для составления легенды на графике. 

# цвета используемые для покраски поверхностей на 3D графике
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

t_range = 6 # интервал изменения t от 0 до t_range

# для каждого найденного решения производим ряд операций
for s in soln :
    dot_stability = 1

    # составляем матрицу Якоби частных производных
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

    yakobi_matrix = np.array([[x11_.real, x12_.real, x13_.real], [x21_.real, x22_.real, x23_.real], \
        [x31_.real, x32_.real, x33_.real]], dtype=complex)

    # находим собственные значения и вектора матрицы Якоби
    wa, va = LA.eig(yakobi_matrix)
    
    print('-----------------------------------------------------------------------------------------------')

    # производим ряд проверок, определяющих устойчивость текущего решения
    # 1. Если хотя бы одно собственное значение  = 0 -> считаем решение точкой смены устойчивости
    # 2. Если хотя бы одно собственное значение  > 0 -> считаем решение неустойчивой точкой равновесия
    # 3. Если     все    собственные   значения  < 0 -> считаем решение асимптотически устойчивой точкой
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
    elif check_lower == len(wa) :
        print('точка V{} {} асимптотически устойчива'.format(soln.index(s)+1, s))

    print('-----------------------------------------------------------------------------------------------')

    stability = ''
    if dot_stability == 0 :
        stability = 'смена устойчивости'
    elif dot_stability == -1 :
        stability = 'неустойчива'
    else :
        stability = 'устойчива'
    
    # заполняем легенду данными об устойчивости решений
    legend.append('V{}- {} {}'.format(soln.index(s)+1, stability, s))

    # инициализация новых элементов массива
    u_t_array.append([]) 

    # для каждого t от 0 до t_range с шагом 1 находим значения Ut и сохраняем результаты в массивы
    for t in range(t_range) :
        x__ = 0

        # для каждого x от 0 до pi с шагом 0.08 находим значения Ut и сохраняем результаты в массивы
        while x__ < pi :

            # подставляем значения решения в U0 с точкой 
            U_dot_0_ = zero_B.subs({U1:s[0], U2:s[1], U3:s[2]})

            # интегрируя U0 с точкой по t получим:
            U0_ = U_dot_0_*t

            # подставляем все найденные значения в исходное уравнение U_t
            U_t_ = complex(U_t.subs({U0:U0_, U1:s[0], U2:s[1], U3:s[2], x:x__}))
            
            # заполняем массивы данных для t и x
            if soln.index(s) == 0 :
                t_array.append(t) # 40 times append t
                x_array.append(x__) # 40 times append 
            
            # для каждого решения записываем значения Ut в отдельный подмассив
            u_t_array[soln.index(s)].append(U_t_.real)

            x__ += 0.08

# инициализируем фигуры под графики
fig = plt.figure()
fig_t_0 = plt.figure()
fig_t_1 = plt.figure()
fig_t_5 = plt.figure()

# инициализируем графики на фигурах
ax = fig.add_subplot(111, projection='3d')
ax1 = fig_t_0.add_subplot(111)
ax2 = fig_t_1.add_subplot(111)
ax3 = fig_t_5.add_subplot(111)

# превращаем одномерные массивы в матрицы для отрисовки в 3D
x_plot = np.reshape(t_array, (t_range, 40))
y_plot = np.reshape(x_array, (t_range, 40))

# для отрисовки каждого решения выполняем ряд действий
for i in range(len(u_t_array)) :

    # превращаем одномерный массив с решениями Ut в матрицу
    z_plot = np.reshape(u_t_array[i], (t_range, 40))

    # отрисовываем 3D график решения Ut
    ax.plot_surface(x_plot, y_plot, z_plot, color=my_colors[i%8])

    # отрисовывываем 2D графики для решения при t = 0,1,5 соответственно
    ax1.plot(y_plot[0], z_plot[0])
    ax2.plot(y_plot[0], z_plot[1])
    ax3.plot(y_plot[0], z_plot[5])

# объявляем отрисовку легенды на 2D графиках
ax1.legend(legend, loc='upper right')
ax2.legend(legend, loc='upper right')
ax3.legend(legend, loc='upper right')

# объявляем названия 2D графикам
ax1.set_title("Ut при t=0")
ax2.set_title("Ut при t=1")
ax3.set_title("Ut при t=5")

# объявляем обозначения числовых осей 3D графика
ax.set_xlabel('T Label')
ax.set_ylabel('X Label')
ax.set_zlabel('U_t Label')

plt.show()