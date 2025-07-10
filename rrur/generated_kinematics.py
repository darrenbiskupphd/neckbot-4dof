# This file is auto-generated. Do not edit manually.
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import optax
import jax.numpy as jnp

@jax.jit
def E1_chain_func(theta_11, theta_12, gamma_11, gamma_12):
    # Note: uses jnp for math functions like sin, cos
    m00 = -0.1*jnp.cos(gamma_11)*jnp.cos(gamma_12)
    m10 = -0.1*jnp.sin(gamma_12) + 0.1*jnp.cos(theta_11) + 0.1*jnp.cos(theta_11 + theta_12) + 0.141
    m20 = 0.1*jnp.sin(gamma_11)*jnp.cos(gamma_12) + 0.1*jnp.sin(theta_11) + 0.1*jnp.sin(theta_11 + theta_12)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def E2_chain_func(theta_21, theta_22, gamma_21, gamma_22):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0.1*jnp.sin(gamma_22) + 0.1*jnp.cos(theta_21) + 0.1*jnp.cos(theta_21 + theta_22) - 0.12
    m10 = -0.1*jnp.cos(gamma_21)*jnp.cos(gamma_22)
    m20 = 0.1*jnp.sin(gamma_21)*jnp.cos(gamma_22) + 0.1*jnp.sin(theta_21) + 0.1*jnp.sin(theta_21 + theta_22)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def E3_chain_func(theta_31, theta_32, gamma_31, gamma_32):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0.1*jnp.cos(gamma_31)*jnp.cos(gamma_32)
    m10 = 0.1*jnp.sin(gamma_32) + 0.1*jnp.cos(theta_31) + 0.1*jnp.cos(theta_31 + theta_32) - 0.141
    m20 = 0.1*jnp.sin(gamma_31)*jnp.cos(gamma_32) + 0.1*jnp.sin(theta_31) + 0.1*jnp.sin(theta_31 + theta_32)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def Q1_func(theta_11):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0.1*jnp.cos(theta_11) + 0.141
    m20 = 0.1*jnp.sin(theta_11)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def Q2_func(theta_21):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0.1*jnp.cos(theta_21) - 0.12
    m10 = 0
    m20 = 0.1*jnp.sin(theta_21)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def Q3_func(theta_31):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0.1*jnp.cos(theta_31) - 0.141
    m20 = 0.1*jnp.sin(theta_31)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def E1_ee_func(alpha, beta, gamma, z):
    # Note: uses jnp for math functions like sin, cos
    m00 = -0.105*jnp.sin(alpha)*jnp.cos(gamma) + 0.105*jnp.sin(beta)*jnp.sin(gamma)*jnp.cos(alpha) + 0.063*jnp.sin(gamma)*jnp.cos(beta)
    m10 = -0.063*jnp.sin(beta) + 0.105*jnp.cos(alpha)*jnp.cos(beta)
    m20 = z + 0.105*jnp.sin(alpha)*jnp.sin(gamma) + 0.105*jnp.sin(beta)*jnp.cos(alpha)*jnp.cos(gamma) + 0.063*jnp.cos(beta)*jnp.cos(gamma)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def E2_ee_func(alpha, beta, gamma, z):
    # Note: uses jnp for math functions like sin, cos
    m00 = -0.055*jnp.sin(alpha)*jnp.sin(beta)*jnp.sin(gamma) + 0.063*jnp.sin(gamma)*jnp.cos(beta) - 0.055*jnp.cos(alpha)*jnp.cos(gamma)
    m10 = -0.055*jnp.sin(alpha)*jnp.cos(beta) - 0.063*jnp.sin(beta)
    m20 = z - 0.055*jnp.sin(alpha)*jnp.sin(beta)*jnp.cos(gamma) + 0.055*jnp.sin(gamma)*jnp.cos(alpha) + 0.063*jnp.cos(beta)*jnp.cos(gamma)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def E3_ee_func(alpha, beta, gamma, z):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0.105*jnp.sin(alpha)*jnp.cos(gamma) - 0.105*jnp.sin(beta)*jnp.sin(gamma)*jnp.cos(alpha) + 0.063*jnp.sin(gamma)*jnp.cos(beta)
    m10 = -0.063*jnp.sin(beta) - 0.105*jnp.cos(alpha)*jnp.cos(beta)
    m20 = z - 0.105*jnp.sin(alpha)*jnp.sin(gamma) - 0.105*jnp.sin(beta)*jnp.cos(alpha)*jnp.cos(gamma) + 0.063*jnp.cos(beta)*jnp.cos(gamma)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def A_func(alpha, beta, gamma, z):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0
    m20 = z
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def S1_func():
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0.141000000000000
    m20 = 0
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def S2_func():
    # Note: uses jnp for math functions like sin, cos
    m00 = -0.120000000000000
    m10 = 0
    m20 = 0
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def S3_func():
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = -0.141000000000000
    m20 = 0
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def U1_func(theta_11, theta_12):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0.1*jnp.cos(theta_11) + 0.1*jnp.cos(theta_11 + theta_12) + 0.141
    m20 = 0.1*jnp.sin(theta_11) + 0.1*jnp.sin(theta_11 + theta_12)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def U2_func(theta_21, theta_22):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0.1*jnp.cos(theta_21) + 0.1*jnp.cos(theta_21 + theta_22) - 0.12
    m10 = 0
    m20 = 0.1*jnp.sin(theta_21) + 0.1*jnp.sin(theta_21 + theta_22)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

@jax.jit
def U3_func(theta_31, theta_32):
    # Note: uses jnp for math functions like sin, cos
    m00 = 0
    m10 = 0.1*jnp.cos(theta_31) + 0.1*jnp.cos(theta_31 + theta_32) - 0.141
    m20 = 0.1*jnp.sin(theta_31) + 0.1*jnp.sin(theta_31 + theta_32)
    return jnp.array([
        [m00],
        [m10],
        [m20],
    ], dtype=float).ravel()

p1 = 0.172
p2 = 0.14
p3 = 0.172


# Constants for the squared lengths of the passive rods
p1_squared = p1**2
p2_squared = p2**2
p3_squared = p3**2

@jax.jit
def objective(ee_params, joint_params, passive_params):
    """
    Kinematics Objective Function.
    Calculates the error vector for a given end-effector and joint configuration.

    Args:
        ee_params (jnp.array): An array containing the end-effector parameters 
                               [alpha, beta, gamma, z].
        joint_params (jnp.array): An array containing all 12 joint angles.

    Returns:
        float: The sum of squared errors for the virtual rods and the end-effector chains.
    """
    alpha, beta, gamma, z = ee_params
    theta_11, theta_21, theta_31, theta_22 = joint_params
    theta_12, theta_32, gamma_11, gamma_12, gamma_21, gamma_22, gamma_31, gamma_32 = passive_params

    # Calculate kinematic quantities based on joint and ee parameters
    U1_val = U1_func(theta_11, theta_12)
    U2_val = U2_func(theta_21, theta_22)
    U3_val = U3_func(theta_31, theta_32)

    E1_chain_val = E1_chain_func(theta_11, theta_12, gamma_11, gamma_12)
    E2_chain_val = E2_chain_func(theta_21, theta_22, gamma_21, gamma_22)
    E3_chain_val = E3_chain_func(theta_31, theta_32, gamma_31, gamma_32)

    A_val = A_func(alpha, beta, gamma, z)
    E1_ee_val = E1_ee_func(alpha, beta, gamma, z)
    E2_ee_val = E2_ee_func(alpha, beta, gamma, z)
    E3_ee_val = E3_ee_func(alpha, beta, gamma, z)

    # Calculate squared distance errors for the passive rods
    p1_squared_calc = jnp.sum((U1_val - A_val)**2)
    p2_squared_calc = jnp.sum((U2_val - A_val)**2)
    p3_squared_calc = jnp.sum((U3_val - A_val)**2)

    # Assemble the 12-element output error_squared vector
    output = jnp.zeros(12)
    output = output.at[0].set(p1_squared_calc - p1_squared)
    output = output.at[1].set(p2_squared_calc - p2_squared)
    output = output.at[2].set(p3_squared_calc - p3_squared)
    output = output.at[3:6].set(E1_chain_val - E1_ee_val)
    output = output.at[6:9].set(E2_chain_val - E2_ee_val)
    output = output.at[9:12].set(E3_chain_val - E3_ee_val)

    return jnp.sum(jnp.square(output))


# Pre-compile gradients for optimization
# Gradient for IK: objective w.r.t. joint and passive parameters
ik_objective_grad = jax.jit(jax.grad(lambda p, e: objective(e, p[:4], p[4:]), argnums=0))

# Gradient for FK: objective w.r.t. end-effector and passive parameters
fk_objective_grad = jax.jit(jax.grad(lambda p, j: objective(p[:4], j, p[4:]), argnums=0))

ik_x0 = jnp.array([jnp.deg2rad(30), jnp.deg2rad(130), jnp.deg2rad(150), jnp.deg2rad(270), jnp.deg2rad(90), jnp.deg2rad(270), jnp.deg2rad(90), 0, jnp.deg2rad(90), 0, jnp.deg2rad(90), 0])
fk_x0 = jnp.array([
    jnp.deg2rad(0),                # alpha (rad)
    jnp.deg2rad(0),                # beta (rad)
    jnp.deg2rad(0),                # gamma (rad)
    0.18,               # z (m)
    jnp.deg2rad(120),   # theta_12 (rad)
    jnp.deg2rad(240),   # theta_32 (rad)
    jnp.deg2rad(90),    # gamma_11 (rad)
    0.0,                # gamma_12 (rad)
    jnp.deg2rad(90),    # gamma_21 (rad)
    0.0,                # gamma_22 (rad)
    jnp.deg2rad(90),    # gamma_31 (rad)
    0.0                 # gamma_32 (rad)
])

@jax.jit
def inverse_kinematics(ee_params, x0=ik_x0, learning_rate=3e-2, tol=1e-6):
    """
    Inverse kinematics solver using optax Adam optimizer.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0)
    params = x0

    def step(params, opt_state):
        grads = ik_objective_grad(params, ee_params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for _ in range(75):
        params, opt_state = step(params, opt_state)

    loss = objective(ee_params, params[:4], params[4:])
    return params, loss

@jax.jit
def forward_kinematics(joint_params, x0=fk_x0, learning_rate=3e-2, tol=1e-6):
    """
    Forward kinematics solver using optax Adam optimizer.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x0)
    params = x0

    def step(params, opt_state):
        grads = fk_objective_grad(params, joint_params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for _ in range(75):
        params, opt_state = step(params, opt_state)

    loss = objective(params[:4], joint_params, params[4:])
    return params, loss
