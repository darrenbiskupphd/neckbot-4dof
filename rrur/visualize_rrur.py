import numpy as np
import plotly.graph_objects as go
from generated_kinematics import *

def visualize_robot(
    # Robot state parameters
    alpha_val, beta_val, gamma_val, z_val,
    theta1_1, theta2_1, theta3_1, theta2_2,
    theta1_2, theta3_2, gamma1_1, gamma1_2, gamma2_1, gamma2_2, gamma3_1, gamma3_2,
    # Optional parameters
    show_figure=False,
    save_html=None,
    showlegend=False,
    width=800,
    height=600,
    x_range=(-.300, .200),
    y_range=(-.250, .250),
    z_range=(-.010, .510),
    title='3D Visualization of Chains and End-Effector',
):
    """
    Create a 3D visualization of the robot with chains and end-effector.
    
    Parameters:
    -----------
    alpha_val, beta_val, gamma_val, z_val : float
        End-effector parameters
    theta1_1, theta2_1, theta3_1, theta2_2 : float
        actuated joint parameters
    theta1_2, theta3_2, gamma1_1, gamma1_2, gamma2_1, gamma2_2, gamma3_1, gamma3_2 : float
        passive joint parameters
    geometric_params : dict, optional
        Dictionary of geometric parameters for the robot
    show_figure : bool, optional
        Whether to display the figure interactively
    save_html : str, optional
        Path to save the HTML file (None = don't save)
    width, height : int, optional
        Figure dimensions in pixels
    x_range, y_range, z_range : tuple, optional
        Axis ranges for the 3D plot
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The 3D figure object
    """
    
    # Evaluate the kinematic functions with the provided parameters
    S1_val = S1_func()
    S2_val = S2_func()
    S3_val = S3_func()

    Q1_val = Q1_func(theta1_1)
    Q2_val = Q2_func(theta2_1)
    Q3_val = Q3_func(theta3_1)

    U1_val = U1_func(theta1_1, theta1_2)
    U2_val = U2_func(theta2_1, theta2_2)
    U3_val = U3_func(theta3_1, theta3_2)

    E1_chain_val = E1_chain_func(theta1_1, theta1_2, gamma1_1, gamma1_2)
    E2_chain_val = E2_chain_func(theta2_1, theta2_2, gamma2_1, gamma2_2)
    E3_chain_val = E3_chain_func(theta3_1, theta3_2, gamma3_1, gamma3_2)

    E1_ee_val = E1_ee_func(alpha_val, beta_val, gamma_val, z_val)
    E2_ee_val = E2_ee_func(alpha_val, beta_val, gamma_val, z_val)
    E3_ee_val = E3_ee_func(alpha_val, beta_val, gamma_val, z_val)
    E_o_val = (E1_ee_val + E3_ee_val)/2

    # Create a 3D figure
    fig = go.Figure()

    # Plot the chains
    fig.add_trace(go.Scatter3d(
        x=[S1_val[0], Q1_val[0], U1_val[0], E1_chain_val[0]],
        y=[S1_val[1], Q1_val[1], U1_val[1], E1_chain_val[1]],
        z=[S1_val[2], Q1_val[2], U1_val[2], E1_chain_val[2]],
        mode='lines+markers',
        line=dict(color='red'),
        name='Chain 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[S2_val[0], Q2_val[0], U2_val[0], E2_chain_val[0]],
        y=[S2_val[1], Q2_val[1], U2_val[1], E2_chain_val[1]],
        z=[S2_val[2], Q2_val[2], U2_val[2], E2_chain_val[2]],
        mode='lines+markers',
        line=dict(color='green'),
        name='Chain 2'
    ))
    fig.add_trace(go.Scatter3d(
        x=[S3_val[0], Q3_val[0], U3_val[0], E3_chain_val[0]],
        y=[S3_val[1], Q3_val[1], U3_val[1], E3_chain_val[1]],
        z=[S3_val[2], Q3_val[2], U3_val[2], E3_chain_val[2]],
        mode='lines+markers',
        line=dict(color='blue'),
        name='Chain 3'
    ))

    # Plot the end-effector points and A
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[z_val],
        mode='markers',
        marker=dict(color='black', size=5, symbol='x'),
        name='A'
    ))
    fig.add_trace(go.Scatter3d(
        x=[E1_ee_val[0]], y=[E1_ee_val[1]], z=[E1_ee_val[2]],
        mode='markers', marker=dict(color='red', size=5), name='E1',
    ))
    fig.add_trace(go.Scatter3d(
        x=[E2_ee_val[0]], y=[E2_ee_val[1]], z=[E2_ee_val[2]],
        mode='markers', marker=dict(color='green', size=5), name='E2',
    ))
    fig.add_trace(go.Scatter3d(
        x=[E3_ee_val[0]], y=[E3_ee_val[1]], z=[E3_ee_val[2]],
        mode='markers', marker=dict(color='blue', size=5), name='E3',
    ))
    fig.add_trace(go.Scatter3d(
        x=[E_o_val[0]], y=[E_o_val[1]], z=[E_o_val[2]],
        mode='markers', marker=dict(color='orange', size=5), name='E_o',
    ))

    # Connect the points to form the prismoid
    # Edge 1
    fig.add_trace(go.Scatter3d(
        x=[E1_ee_val[0], E2_ee_val[0]],
        y=[E1_ee_val[1], E2_ee_val[1]],
        z=[E1_ee_val[2], E2_ee_val[2]],
        mode='lines', line=dict(color='black', dash='solid'),
        name='Edge 1', showlegend=False
    ))
    # Edge 2
    fig.add_trace(go.Scatter3d(
        x=[E2_ee_val[0], E3_ee_val[0]],
        y=[E2_ee_val[1], E3_ee_val[1]],
        z=[E2_ee_val[2], E3_ee_val[2]],
        mode='lines', line=dict(color='black', dash='solid'),
        name='Edge 2', showlegend=False
    ))
    # Edge 3
    fig.add_trace(go.Scatter3d(
        x=[E3_ee_val[0], E1_ee_val[0]],
        y=[E3_ee_val[1], E1_ee_val[1]],
        z=[E3_ee_val[2], E1_ee_val[2]],
        mode='lines', line=dict(color='black', dash='solid'),
        name='Edge 3', showlegend=False
    ))
    # A to E1
    fig.add_trace(go.Scatter3d(
        x=[0, E1_ee_val[0]], y=[0, E1_ee_val[1]], z=[z_val, E1_ee_val[2]],
        mode='lines', line=dict(color='black', dash='dash'), name='A to E1'
    ))
    # A to E2
    fig.add_trace(go.Scatter3d(
        x=[0, E2_ee_val[0]], y=[0, E2_ee_val[1]], z=[z_val, E2_ee_val[2]],
        mode='lines', line=dict(color='black', dash='dash'), name='A to E2'
    ))
    # A to E3
    fig.add_trace(go.Scatter3d(
        x=[0, E3_ee_val[0]], y=[0, E3_ee_val[1]], z=[z_val, E3_ee_val[2]],
        mode='lines', line=dict(color='black', dash='dash'), name='A to E3'
    ))

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title=title,
        width=width,
        height=height,
        showlegend=showlegend,
    )

    # Show the figure if requested
    if show_figure:
        fig.show()

    # Save the figure as an HTML file if a path is provided
    if save_html:
        fig.write_html(save_html)

    return fig