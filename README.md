# neckbot-4dof: 3-RPUR/RRUR Parallel Neck Brace Mechanism for Human Rehabilitation

This repository contains design files (soon), control code, and analysis scripts of a **4-DOF parallel neck brace mechanism**. This assistive robotic mechanism is designed to provide neck support while allowing human-like head motion.

---
## 📦 Dependencies
The following python packages are required:
- `sympy`
- `scipy`
- `numba`
- `plotly`
 - `kaleido`
- `imageio`

install them with
```bash
pip install scipy numba plotly
```
---

## 🗂️ Key Files
- `neck_brace_kinematics.py`
- `vizualize_robot.py`

### `neck_brace_kinematics.py`.
You can should import functions `forward_kinematics` and `inverse_kinematics`

#### `forward_kinematics(...)`
```python
forward_kinematics(
    d1_val, d2_val, d3_val,         # Actuator displacements (e.g., prismatic joint extensions)
    theta2_val,                     # Internal joint angle
    initial_guess=np.array([...]), # Initial guess for the solver
    xtol=1e-3                       # Solver convergence tolerance
)
```

- `d1_val`, `d2_val`, `d3_val`: Linear displacements of three actuators.
- `theta2_val`: posterior chain revolute joint angle
- `initial_guess`: 12-element NumPy array used as the initial guess for nonlinear solving.
- `xtol`: Tolerance for solver convergence.

#### Returns
- A NumPy array of internal configuration variables including:
  - first 4 variables End-effector orientation and position (`alpha`, `beta`, `gamma`, `z`)
  - remaining variables are passive joint angles

#### `inverse_kinematics(...)`
```python
inverse_kinematics(
    alpha_val, beta_val, gamma_val, z_val # Orientation angles (Euler Body-312) and z-height of "A"
    initial_guess=np.array([...]), # Initial guess for solver
    xtol=1e-5                      # Solver convergence tolerance
)
```

- `alpha_val`, `beta_val`, `gamma_val`: Desired orientation of the end-effector in Euler angles (Body-312 rotation).
- `z_val`: Desired height of the end-effector virtual point A.
- `initial_guess`: 12-element NumPy array with actuator and joint guesses.
- `xtol`: Tolerance for solver convergence.

#### Returns
- A NumPy array of actuator and joint variables including:
  - `d1`, `d2`, `d3`: Actuator displacements
  - `theta2`: posterior chain revolute joint angle
  - Passive joint angles (`theta1`, `theta3`, etc.)

#### `visualize_robot(...)`
```python
visualize_robot(
    alpha_val, beta_val, gamma_val, z_val,
    d1_val, d2_val, d3_val, theta2_val,
    theta1_val, gamma1_1_val, gamma2_1_val,
    gamma1_2_val, gamma2_2_val,
    theta3_val, gamma1_3_val, gamma2_3_val,
    geometric_params=None,
    show_figure=False,
    save_html=None,
    width=800,
    height=600,
    x_range=(-.210, .210),
    y_range=(-.110, .310),
    z_range=(-.010, .410),
    title='3D Visualization of Chains and End-Effector',
)
```

- `alpha_val`, `beta_val`, `gamma_val`, `z_val`: End-effector pose.
- `d1_val`, `d2_val`, `d3_val`, `theta2_val`: Actuator and joint variables.
- `theta1_val`, `theta3_val`, `gamma*_val`: Passive joint angles.
- `geometric_params`: Optional dictionary of robot geometry.
- `show_figure`: Whether to show the interactive Plotly figure.
- `save_html`: Path to save the figure as an HTML file.
- `width`, `height`: Plot dimensions in pixels.
- `x_range`, `y_range`, `z_range`: Axis ranges for the 3D plot.
- `title`: Title of the 3D plot.

#### Returns
A `plotly.graph_objects.Figure` object representing the robot in 3D space.

