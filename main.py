from cmath import cos as cos
from cmath import sin as sin
from math import pi
import math
from sympy import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# объявление переменных, которые будут использоваться в расчетах
x, U0, U1, U2, U3, U4, U_dot_0, U_dot_1, U_dot_2, U_dot_3, U_dot_4, b, c = symbols('x U0 U1 U2 U3 U4 U_dot_0 U_dot_1 U_dot_2 U_dot_3 U_dot_4 b c')

# определение b, c и собственной функции
b_ = 2 # с _ т.к. эти символы используются в уравнении
c_ = 1 # -||-

def own_function(n,x) :
    return cos(n*x)

# определение изначальной задачи Ut
U_t = U0/2 + U1*own_function(1,x) + U2*own_function(2,x) + U3*own_function(3,x) + U4*own_function(4,x)
U_dot = U_dot_0/2 + U_dot_1*own_function(1,x) + U_dot_2*own_function(2,x) + U_dot_3*own_function(3,x) + U_dot_4*own_function(4,x)

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
print("Система дифференциальных уравнений:")
#нахождение 1го интеграла
first_differential_equation = integrate(B, (x, 0, pi))
print('{} = {}'.format(U_dot_0, first_differential_equation))

#нахождение 2го интеграла
second_differential_equation = integrate(B*own_function(1,x), (x, 0, pi))
print('{} = {}'.format(U_dot_1, second_differential_equation))

#нахождение 3го интеграла
third_differential_equation = integrate(B*own_function(2,x), (x, 0, pi))
print('{} = {}'.format(U_dot_2, third_differential_equation))

#нахождение 4го интеграла
fourth_differential_equation = integrate(B*own_function(3,x), (x, 0, pi))
print('{} = {}'.format(U_dot_3, fourth_differential_equation))

#нахождение 5го интеграла
fifth_differential_equation = integrate(B*own_function(4,x), (x, 0, pi))
print('{} = {}'.format(U_dot_4, fifth_differential_equation))

# далее решаем полученную систему дифференциальных уравнений
try :
    system_solutions = solve([second_differential_equation.subs(b, b_), third_differential_equation.subs(b, b_), fourth_differential_equation.subs(b, b_)], U1, U2, U3)

    # все найденные решения системы переводим решения из символьной формы в числовую (в число с плавающей точкой)
    solutions = [tuple(sol_value.evalf() for sol_value in sol) for sol in system_solutions]
    print('\nРешения системы уравнений при b={}:\n{}'.format(b_, solutions))
except Exception as e :
    print(e)

# определяем необходимые структуры для сохранения решений
t_array = [] # массив содержащий значения t ( 6 раз по 40 значений 0 <= t <= 6)
x_array = [] # массив содержащий значения x ( 6 раз по 40 значений 0 <= x <= pi)
ut_array = [] # многомерный массив, который содержит массивы решений для Ut. Размерность массива = кол-ву решений. Каждый массив состоит из 6 подмассивов по 40 значений Ut
legend = [] # массив содержащий информацию об устойчивости решения, собственно решения и его индекса. Используется для составления легенды на графике. 

# цвета используемые для покраски поверхностей на 3D графике
plot_colors = [
    'b',#	голубой
    'g',#	зелёный
    'r',#	красный
    'c',#	циановый
    'm',#	пурпурный
    'y',#	желтый
    'k',#	черный
    'w' #	белый
]

t_range = 6 # интервал изменения t от 0 до t_range
x_increment_size = 0.08 # значение шага x от 0 до pi || чем меньше тем более гладкие линии (значение 0.08 оптимально) 
shape_size = math.ceil(pi/x_increment_size)

print("\nАнализ устойчивости найденных решений:")

# для каждого найденного решения производим ряд операций
for solution in solutions :
    dot_stability = 1

    # составляем матрицу Якоби частных производных
    x11 = diff(second_differential_equation.subs(b,b_),U1)
    x11_ = complex(x11.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x12 = diff(second_differential_equation.subs(b,b_),U2)
    x12_ = complex(x12.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x13 = diff(second_differential_equation.subs(b,b_),U3)
    x13_ = complex(x13.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x21 = diff(third_differential_equation.subs(b,b_),U1)
    x21_ = complex(x21.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x22 = diff(third_differential_equation.subs(b,b_),U2)
    x22_ = complex(x22.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x23 = diff(third_differential_equation.subs(b,b_),U3)
    x23_ = complex(x23.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x31 = diff(fourth_differential_equation.subs(b,b_),U1)
    x31_ = complex(x31.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x32 = diff(fourth_differential_equation.subs(b,b_),U2)
    x32_ = complex(x32.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    x33 = diff(fourth_differential_equation.subs(b,b_),U3)
    x33_ = complex(x33.subs({U1:solution[0], U2:solution[1], U3:solution[2]}))

    yakobi_matrix = np.array([[x11_.real, x12_.real, x13_.real], [x21_.real, x22_.real, x23_.real], \
        [x31_.real, x32_.real, x33_.real]], dtype=complex)

    # находим собственные значения и вектора матрицы Якоби
    eigenvalues, eigenvectors = LA.eig(yakobi_matrix)

    # производим ряд проверок, определяющих устойчивость текущего решения
    # 1. Если хотя бы одно собственное значение  = 0 -> считаем решение точкой смены устойчивости
    # 2. Если хотя бы одно собственное значение  > 0 -> считаем решение неустойчивой точкой равновесия
    # 3. Если     все    собственные   значения  < 0 -> считаем решение асимптотически устойчивой точкой
    check_lower = 0
    check_upper = 0
    check_zero = 0
    for eigenvalue in eigenvalues :
        if round(eigenvalue.real,8) == 0 :
            check_zero += 1
        if eigenvalue.real < 0 :
            check_lower += 1
        if eigenvalue.real > 0 :
            check_upper += 1
    if check_zero != 0 :
        dot_stability = 0
        print("Точка V{} {} явялется точкой смены устойчивости при b = {} и  собственном значении = {}".format(solutions.index(solution)+1, solution, b_, eigenvalues))
    elif check_upper != 0 :
        dot_stability = -1
        print('Точка V{} {} равновесия не устойчива'.format(solutions.index(solution)+1, solution))
    elif check_lower == len(eigenvalues) :
        print('Точка V{} {} асимптотически устойчива'.format(solutions.index(solution)+1, solution))

    stability = ''
    if dot_stability == 0 :
        stability = 'смена устойчивости'
    elif dot_stability == -1 :
        stability = 'неустойчива'
    else :
        stability = 'устойчива'
    
    # заполняем легенду данными об устойчивости решений
    legend.append('V{}- {} {}'.format(solutions.index(solution)+1, stability, solution))

    # инициализация новых элементов массива
    ut_array.append([]) 

    # для каждого t от 0 до t_range с шагом 1 находим значения Ut и сохраняем результаты в массивы
    for t in range(t_range) :
        x_ = 0

        # для каждого x от 0 до pi с шагом 0.08 находим значения Ut и сохраняем результаты в массивы
        while x_ < pi :

            # подставляем значения решения в U0 с точкой 
            U_dot_0_ = first_differential_equation.subs({U1:solution[0], U2:solution[1], U3:solution[2]})

            # интегрируя U0 с точкой по t получим:
            U0_ = U_dot_0_*t

            # подставляем все найденные значения в исходное уравнение U_t
            U_t_ = complex(U_t.subs({U0:U0_, U1:solution[0], U2:solution[1], U3:solution[2], x:x_}))
            
            # заполняем массивы данных для t и x
            if solutions.index(solution) == 0 :
                t_array.append(t) # 40 раз добавляем значение t
                x_array.append(x_) # 40 раз добавляем значения x
            
            # для каждого решения записываем значения Ut в отдельный подмассив
            ut_array[solutions.index(solution)].append(U_t_.real)
            x_ += x_increment_size

# инициализируем фигуры под графики
figure_3d_plot = plt.figure()
figure_2d_t0   = plt.figure()
figure_2d_t1   = plt.figure()
figure_2d_t5   = plt.figure()

# инициализируем графики на фигурах
axes_3d_plot = figure_3d_plot.add_subplot(111, projection='3d')
axes_2d_t0   = figure_2d_t0.add_subplot(111)
axes_2d_t1   = figure_2d_t1.add_subplot(111)
axes_2d_t5   = figure_2d_t5.add_subplot(111)

# превращаем одномерные массивы в матрицы для отрисовки в 3D
t_axis = np.reshape(t_array, (t_range, shape_size)) # x ось по умолчанию
x_axis = np.reshape(x_array, (t_range, shape_size)) # y ось по умолчанию

# для отрисовки каждого решения выполняем ряд действий
for index in range(len(ut_array)) :

    # превращаем одномерный массив с решениями Ut в матрицу
    ut_axis = np.reshape(ut_array[index], (t_range, shape_size)) # z ось по умолчанию

    # отрисовываем 3D график решения Ut
    surf = axes_3d_plot.plot_surface(t_axis, x_axis, ut_axis, color=plot_colors[index%8], label=legend[index])

    # эта конструкция помогает решить проблему с цветами в легенде для 3D графика
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    # отрисовывываем 2D графики для решения при t = 0,1,5 соответственно
    axes_2d_t0.plot(x_axis[0], ut_axis[0])
    axes_2d_t1.plot(x_axis[0], ut_axis[1])
    axes_2d_t5.plot(x_axis[0], ut_axis[5])

# объявляем отрисовку легенды на 2D графиках
axes_2d_t0.legend(legend, loc='upper right')
axes_2d_t1.legend(legend, loc='upper right')
axes_2d_t5.legend(legend, loc='upper right')

# объявляем названия 2D графикам
axes_2d_t0.set_title("График решений Ut при t = 0 и 0 <= x <= Pi")
axes_2d_t1.set_title("График решений Ut при t = 1 и 0 <= x <= Pi")
axes_2d_t5.set_title("График решений Ut при t = 5 и 0 <= x <= Pi")

# объявляем обозначения числовых осей 3D графика
axes_3d_plot.set_xlabel('ось t')
axes_3d_plot.set_ylabel('ось x')
axes_3d_plot.set_zlabel('ось Ut')

# объявляем название 3D графика
axes_3d_plot.set_title(f"График решений Ut при 0 <= t <= {t_range - 1} и 0 <= x <= Pi")
axes_3d_plot.legend(loc='best')

plt.show()