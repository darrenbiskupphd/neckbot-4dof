import numpy as np
from sympy import *
from scipy.optimize import fsolve
import numba as nb

# Constants
l1, l2, l3 = symbols('l_1 l_2 l_3')
d1, d2, d3 = symbols('d_1 d_2 d_3')

D, B = symbols('D B')

theta1, theta2, theta3 = symbols('theta_1 theta_2 theta_3')
gamma1_1, gamma1_2, gamma1_3 = symbols('gamma_1_1 gamma_1_2 gamma_1_3') # gamma a chain b
gamma2_1, gamma2_2, gamma2_3 = symbols('gamma_2_1 gamma_2_2 gamma_2_3')

# Position of chain base frame expressed in inertial frame
S1 = Matrix([D, 0, 0])
S2 = Matrix([0, B, 0])
S3 = Matrix([-D, 0, 0])

Q1 = S1 + d1*Matrix([-cos(theta1), 0, sin(theta1)])
E1_chain = Q1 + l1*Matrix([-cos(gamma2_1)*cos(gamma1_1), -sin(gamma2_1),cos(gamma2_1)*sin(gamma1_1) ])

Q3 = S3 + d3*Matrix([-cos(theta3), 0, sin(theta3)])
E3_chain = Q3 + l3*Matrix([cos(gamma2_3)*cos(gamma1_3), sin(gamma2_3),cos(gamma2_3)*sin(gamma1_3)])

Q2 = S2 + d2*Matrix([0,cos(theta2),sin(theta2)])
E2_chain = Q2 + l2*Matrix([sin(gamma2_2), -cos(gamma2_2)*cos(gamma1_2), cos(gamma2_2)*sin(gamma1_2)])


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
E1_ee = E_o + R * Matrix([length / 2, 0, 0])
E3_ee = E_o + R * Matrix([-length / 2, 0, 0])
E2_ee = E_o + R * Matrix([0, width, 0])


p1 = .172 # fixed distance between Q1 and A
p2 = .140 # fixed distance between Q2 and A
p3 = p1 # fixed distance between Q3 and A

# now, solve kinematics
geometric_params = {     
    l1: .100, # m     
    l2: .100, # m     
    l3: .100, # m     
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
E1_ee_geo = E1_ee.subs(geometric_params)
E2_ee_geo = E2_ee.subs(geometric_params)
E3_ee_geo = E3_ee.subs(geometric_params)
A_geo = A.subs(geometric_params)

# Lambdify once with joint variables as parameters
E1_chain_func = lambdify((theta1, d1, gamma1_1, gamma2_1), E1_chain_geo, 'numpy')
E2_chain_func = lambdify((theta2, d2, gamma1_2, gamma2_2), E2_chain_geo, 'numpy')
E3_chain_func = lambdify((theta3, d3, gamma1_3, gamma2_3), E3_chain_geo, 'numpy')

Q1_func = lambdify((theta1, d1), Q1_geo, 'numpy')
Q2_func = lambdify((theta2, d2), Q2_geo, 'numpy')
Q3_func = lambdify((theta3, d3), Q3_geo, 'numpy')

E1_ee_func = lambdify((alpha, beta, gamma, z), E1_ee_geo, 'numpy')
E2_ee_func = lambdify((alpha, beta, gamma, z), E2_ee_geo, 'numpy')
E3_ee_func = lambdify((alpha, beta, gamma, z), E3_ee_geo, 'numpy')
A_func = lambdify((alpha, beta, gamma, z), A_geo, 'numpy')

def evaluate_chain_fk(d1_val, d2_val, d3_val, theta2_val,
                      theta1_val, gamma1_1_val, gamma2_1_val, gamma1_2_val, gamma2_2_val, 
                      theta3_val, gamma1_3_val, gamma2_3_val):
    
    E1_chain_val = E1_chain_func(theta1_val, d1_val, gamma1_1_val, gamma2_1_val)
    E2_chain_val = E2_chain_func(theta2_val, d2_val, gamma1_2_val, gamma2_2_val)
    E3_chain_val = E3_chain_func(theta3_val, d3_val, gamma1_3_val, gamma2_3_val)
    Q1_val = Q1_func(theta1_val, d1_val)
    Q2_val = Q2_func(theta2_val, d2_val)
    Q3_val = Q3_func(theta3_val, d3_val)
    return E1_chain_val, E2_chain_val, E3_chain_val, Q1_val, Q2_val, Q3_val

def evaluate_ee_fk(alpha_val, beta_val, gamma_val, z_val):
    E1_ee_val = E1_ee_func(alpha_val, beta_val, gamma_val, z_val)
    E2_ee_val = E2_ee_func(alpha_val, beta_val, gamma_val, z_val)
    E3_ee_val = E3_ee_func(alpha_val, beta_val, gamma_val, z_val)
    A_val = A_func(alpha_val, beta_val, gamma_val, z_val)
    return E1_ee_val, E2_ee_val, E3_ee_val, A_val



# 1. JIT compile the calculation-intensive part of obj_fcn
@nb.njit
def calculate_objective(Q1_val, Q2_val, Q3_val, A_val, 
                        E1_chain_val, E2_chain_val, E3_chain_val,
                        E1_ee_val, E2_ee_val, E3_ee_val, p1_squared, p2_squared, p3_squared):
    p1_calc = (Q1_val - A_val)**2
    p2_calc = (Q2_val - A_val)**2
    p3_calc = (Q3_val - A_val)**2

    output = np.zeros(12)
    output[0:3] = np.array([np.sum(p1_calc) - p1_squared,
                          np.sum(p2_calc) - p2_squared,
                          np.sum(p3_calc) - p3_squared])
    output[3:6] = (E1_chain_val - E1_ee_val).ravel()
    output[6:9] = (E2_chain_val - E2_ee_val).ravel()
    output[9:12] = (E3_chain_val - E3_ee_val).ravel()
    return output

def forward_kinematics(d1_val, d2_val, d3_val, theta2_val, initial_guess=np.array([0.1,0.1,0.1,0.1602,np.deg2rad(100),0,0,0,0,np.deg2rad(80),0,0]), xtol=1e-3):
    # Precompute squared values
    p1_squared, p2_squared, p3_squared = p1**2, p2**2, p3**2
    
    def obj_fcn(vars):
        alpha, beta, gamma, z, theta1_val, gamma1_1_val, gamma2_1_val, gamma1_2_val, gamma2_2_val, theta3_val, gamma1_3_val, gamma2_3_val = vars
        
        # These function calls can't be jitted directly due to SymPy lambdify
        # but lambdified functions are somewhat compiled/optimized
        E1_ee_val, E2_ee_val, E3_ee_val, A_val = evaluate_ee_fk(alpha, beta, gamma, z)
        E1_chain_val, E2_chain_val, E3_chain_val, Q1_val, Q2_val, Q3_val = evaluate_chain_fk(
            d1_val, d2_val, d3_val, theta2_val,
            theta1_val, gamma1_1_val, gamma2_1_val,
            gamma1_2_val, gamma2_2_val,
            theta3_val, gamma1_3_val, gamma2_3_val)
        
        # Use jitted calculation for the computationally intensive part
        return calculate_objective(Q1_val, Q2_val, Q3_val, A_val, 
                                  E1_chain_val, E2_chain_val, E3_chain_val,
                                  E1_ee_val, E2_ee_val, E3_ee_val,
                                  p1_squared, p2_squared, p3_squared)

    soln = fsolve(obj_fcn, initial_guess, xtol=xtol)
    return soln

def inverse_kinematics(alpha_val, beta_val, gamma_val, z_val, initial_guess=np.array([.150, .150, 0.150, np.pi/2, np.pi/2, 0, 0, 0, 0, np.pi/2, 0, 0]), xtol=1e-5):
    E1_ee_val, E2_ee_val, E3_ee_val, A_val = evaluate_ee_fk(alpha_val, beta_val, gamma_val, z_val)
    
    p1_squared, p2_squared, p3_squared = p1**2, p2**2, p3**2
    def constraints(vars):
        d1, d2, d3, theta2,theta1, gamma1_1, gamma2_1, gamma1_2, gamma2_2, theta3, gamma1_3, gamma2_3 = vars
        
        E1_chain_val, E2_chain_val, E3_chain_val, Q1_val, Q2_val, Q3_val = evaluate_chain_fk(
            d1, d2, d3, theta2, theta1, gamma1_1, gamma2_1, gamma1_2, gamma2_2, theta3, gamma1_3, gamma2_3)
        return calculate_objective(Q1_val, Q2_val, Q3_val, A_val, 
                                  E1_chain_val, E2_chain_val, E3_chain_val,
                                  E1_ee_val, E2_ee_val, E3_ee_val,
                                  p1_squared, p2_squared, p3_squared)

    
    soln = fsolve(constraints, initial_guess, xtol=xtol)
    return soln
