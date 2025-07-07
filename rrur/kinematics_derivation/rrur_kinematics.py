import numpy as np
from sympy import *
from scipy.optimize import fsolve

# Constants
l_11, l_12, l_13 = symbols('l_{11} l_{12} l_{13}') # chain a length b
l_21, l_22, l_23 = symbols('l_{21} l_{22} l_{23}')
l_31, l_32, l_33 = symbols('l_{31} l_{32} l_{33}')

D, B = symbols('D B')

theta1_1, theta1_2 = symbols('theta_{11} theta_{12}') # chain a theta b
theta2_1, theta2_2 = symbols('theta_{21} theta_{22}')
theta3_1, theta3_2 = symbols('theta_{31} theta_{32}')

gamma1_1, gamma1_2 = symbols('gamma_{11} gamma_{12}') # chain a gamma b
gamma2_1, gamma2_2 = symbols('gamma_{21} gamma_{22}')
gamma3_1, gamma3_2 = symbols('gamma_{31} gamma_{32}')

# Position of chain base frame expressed in inertial frame
S1 = Matrix([0, D, 0])
S2 = Matrix([-B, 0, 0])
S3 = Matrix([0, -D, 0])

Q1 = S1 + l_11*Matrix([0, cos(theta1_1),  sin(theta1_1)])
Q2 = S2 + l_21*Matrix([cos(theta2_1), 0,  sin(theta2_1)])
Q3 = S3 + l_31*Matrix([0, cos(theta3_1),  sin(theta3_1)])

U1 = Q1 + l_12*Matrix([0, cos(theta1_1 + theta1_2),  sin(theta1_1 + theta1_2)])
U2 = Q2 + l_22*Matrix([cos(theta2_1 + theta2_2), 0,  sin(theta2_1 + theta2_2)])
U3 = Q3 + l_32*Matrix([0, cos(theta3_1 + theta3_2),  sin(theta3_1 + theta3_2)])

E1_chain = U1 + l_13*Matrix([-cos(gamma1_2)*cos(gamma1_1), -sin(gamma1_2),cos(gamma1_2)*sin(gamma1_1)])
E2_chain = U2 + l_23*Matrix([sin(gamma2_2), -cos(gamma2_2)*cos(gamma2_1), cos(gamma2_2)*sin(gamma2_1)])
E3_chain = U3 + l_33*Matrix([cos(gamma3_2)*cos(gamma3_1), sin(gamma3_2),cos(gamma3_2)*sin(gamma3_1)])


# Prismoid dimensions
length, width, height = symbols('length width height')
z = symbols('z') #distance from o to A
alpha, beta, gamma = symbols('alpha beta gamma')

A = Matrix([0,0,z])

# Rotation matrices for 3-1-2 rotation (Z,X,Y)
R_beta = Matrix([
    [1, 0, 0],
    [0, cos(beta), -sin(beta)],
    [0, sin(beta), cos(beta)]
])

R_gamma = Matrix([
    [cos(gamma), 0, sin(gamma)],
    [0, 1, 0],
    [-sin(gamma), 0, cos(gamma)]
])

R_alpha = Matrix([
    [cos(alpha), -sin(alpha), 0],
    [sin(alpha), cos(alpha), 0],
    [0, 0, 1]
])

# Rotation to express vector in base frame to vector in rotated frame
R = R_gamma * R_beta * R_alpha
E_o = A + R*Matrix([0,0,height])

# Define the points of the prismoid
E1_ee = E_o + R * Matrix([0, length / 2, 0])
E3_ee = E_o + R * Matrix([0, -length / 2, 0])
E2_ee = E_o + R * Matrix([-width,0, 0])


p1 = .172 # fixed distance between Q1 and A
p2 = .140 # fixed distance between Q2 and A
p3 = p1 # fixed distance between Q3 and A

# now, solve kinematics
geometric_params = {     
    l_11: .100, # m     
    l_12: .100, # m     
    l_13: .100, # m
    l_21: .100, # m     
    l_22: .100, # m     
    l_23: .100, # m     
    l_31: .100, # m     
    l_32: .100, # m     
    l_33: .100, # m     
    D: .141,     
    B: .120,     
    # width length height of prismoid end effector     
    width: .055, # m     
    length: .210, # m     
    height: .063, # m 
}

# Apply only geometric parameters to keep joint variables symbolic
E1_chain_geo = E1_chain.subs(geometric_params)
E2_chain_geo = E2_chain.subs(geometric_params)
E3_chain_geo = E3_chain.subs(geometric_params)
Q1_geo = Q1.subs(geometric_params)
Q2_geo = Q2.subs(geometric_params)
Q3_geo = Q3.subs(geometric_params)
U1_geo = U1.subs(geometric_params)
U2_geo = U2.subs(geometric_params)
U3_geo = U3.subs(geometric_params)
E1_ee_geo = E1_ee.subs(geometric_params)
E2_ee_geo = E2_ee.subs(geometric_params)
E3_ee_geo = E3_ee.subs(geometric_params)
A_geo = A.subs(geometric_params)
S1_geo = S1.subs(geometric_params)
S2_geo = S2.subs(geometric_params)
S3_geo = S3.subs(geometric_params)




### Jacobian Calculation
## 12 constraint equations
f = Matrix.vstack(
    E1_ee - E1_chain, 
    E2_ee - E2_chain, 
    E3_ee - E3_chain, 
    Matrix([(Q1 - A).dot(Q1 - A) - p1**2]),
    Matrix([(Q2 - A).dot(Q2 - A) - p2**2]),
    Matrix([(Q3 - A).dot(Q3 - A) - p3**2])
)

q = Matrix([theta1_1,theta2_1,theta3_1, theta2_2, theta1_2, theta3_2, gamma1_1, gamma1_2, gamma2_1, gamma2_2, gamma3_1, gamma3_2, alpha, beta, gamma, z])

full_J = f.jacobian(q)

C = full_J.subs(geometric_params)[:, :4]
C_star = full_J.subs(geometric_params)[:, 4:]

# Lambdify C and C_star for numerical evaluation
C_func = lambdify(q, C, modules='numpy')
C_star_func = lambdify(q, C_star, modules='numpy')
#####