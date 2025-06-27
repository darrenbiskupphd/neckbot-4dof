import numpy as np
import plotly.graph_objects as go
from neck_brace_kinematics import (
    S1, S2, S3, 
    evaluate_ee_fk, evaluate_chain_fk,
    geometric_params as default_geometric_params
)

def visualize_robot(
    # Robot state parameters
    alpha_val, beta_val, gamma_val, z_val,
    d1_val, d2_val, d3_val, theta2_val,
    theta1_val, gamma1_1_val, gamma2_1_val,
    gamma1_2_val, gamma2_2_val,
    theta3_val, gamma1_3_val, gamma2_3_val,
    # Optional parameters
    geometric_params=None,
    show_figure=False,
    save_html=None,
    width=800,
    height=600,
    x_range=(-.210, .210),
    y_range=(-.110, .310),
    z_range=(-.010, .410),
    title='3D Visualization of Chains and End-Effector',
):
    """
    Create a 3D visualization of the robot with chains and end-effector.
    
    Parameters:
    -----------
    alpha_val, beta_val, gamma_val, z_val : float
        End-effector parameters
    d1_val, d2_val, d3_val, theta2_val, etc. : float
        Chain kinematic parameters
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
    
    # Use provided geometric parameters or defaults
    if geometric_params is None:
        geometric_params = default_geometric_params
    
    # Perform forward kinematics
    E1_ee_val, E2_ee_val, E3_ee_val, A_val = evaluate_ee_fk(alpha_val, beta_val, gamma_val, z_val)
    E1_chain_val, E2_chain_val, E3_chain_val, Q1_val, Q2_val, Q3_val = evaluate_chain_fk(
        d1_val, d2_val, d3_val, theta2_val,
        theta1_val, gamma1_1_val, gamma2_1_val,
        gamma1_2_val, gamma2_2_val,
        theta3_val, gamma1_3_val, gamma2_3_val)

    # Convert to numpy arrays and flatten
    Q1_val = Q1_val.flatten().astype(np.float64)
    Q2_val = Q2_val.flatten().astype(np.float64)
    Q3_val = Q3_val.flatten().astype(np.float64)
    E1_chain_val = E1_chain_val.flatten().astype(np.float64)
    E2_chain_val = E2_chain_val.flatten().astype(np.float64)
    E3_chain_val = E3_chain_val.flatten().astype(np.float64)
    E1_ee_val = E1_ee_val.flatten().astype(np.float64)
    E2_ee_val = E2_ee_val.flatten().astype(np.float64)
    E3_ee_val = E3_ee_val.flatten().astype(np.float64)
    E_o_val = 0.5*(E1_ee_val + E3_ee_val)
    S1_val = np.array(S1.subs(geometric_params)).astype(np.float64).flatten()
    S2_val = np.array(S2.subs(geometric_params)).astype(np.float64).flatten()
    S3_val = np.array(S3.subs(geometric_params)).astype(np.float64).flatten()

    


    # Create a 3D figure
    fig = go.Figure()

    # Plot the chains
    fig.add_trace(go.Scatter3d(
        x=[S1_val[0], Q1_val[0], E1_chain_val[0]],
        y=[S1_val[1], Q1_val[1], E1_chain_val[1]],
        z=[S1_val[2], Q1_val[2], E1_chain_val[2]],
        mode='lines+markers',
        line=dict(color='red'),
        name='Chain 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[S2_val[0], Q2_val[0], E2_chain_val[0]],
        y=[S2_val[1], Q2_val[1], E2_chain_val[1]],
        z=[S2_val[2], Q2_val[2], E2_chain_val[2]],
        mode='lines+markers',
        line=dict(color='green'),
        name='Chain 2'
    ))
    fig.add_trace(go.Scatter3d(
        x=[S3_val[0], Q3_val[0], E3_chain_val[0]],
        y=[S3_val[1], Q3_val[1], E3_chain_val[1]],
        z=[S3_val[2], Q3_val[2], E3_chain_val[2]],
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
        mode='markers', marker=dict(color='red', size=5), name='E1'
    ))
    fig.add_trace(go.Scatter3d(
        x=[E2_ee_val[0]], y=[E2_ee_val[1]], z=[E2_ee_val[2]],
        mode='markers', marker=dict(color='green', size=5), name='E2'
    ))
    fig.add_trace(go.Scatter3d(
        x=[E3_ee_val[0]], y=[E3_ee_val[1]], z=[E3_ee_val[2]],
        mode='markers', marker=dict(color='blue', size=5), name='E3'
    ))
    fig.add_trace(go.Scatter3d(
        x=[E_o_val[0]], y=[E_o_val[1]], z=[E_o_val[2]],
        mode='markers', marker=dict(color='orange', size=5), name='E_o'
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
        height=height
    )

    # Show the figure if requested
    if show_figure:
        fig.show()

    # Save the figure as an HTML file if a path is provided
    if save_html:
        fig.write_html(save_html)

    return fig