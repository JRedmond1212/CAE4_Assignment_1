import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import os

# =========================================================
# Constants and colours
# =========================================================

TEMPLATE_FILENAME = "blade_bezier_assignment_template.xlsx"

COLOR_ROOT_C1 = "#1f77b4"   # blue
COLOR_ROOT_C2 = "#d62728"   # red
COLOR_TIP_C1  = "#2ca02c"   # green
COLOR_TIP_C2  = "#ff7f0e"   # orange
COLOR_HUB     = "#9467bd"   # purple

# CFD domain factors (chord-based extents).
# Adjust if your assignment uses different multipliers.
CFD_X_UPSTREAM_CHORDS   = 30.0   # domain extends this many chords upstream of rotor plane
CFD_X_DOWNSTREAM_CHORDS = 120.0  # domain extends this many chords downstream
CFD_RADIAL_CHORDS       = 30.0   # radial / vertical extents in multiples of chord

# =========================================================
# Maths helpers
# =========================================================

def binomial_coeff(n, k):
    """Binomial coefficient C(n,k)."""
    from math import comb
    return comb(n, k)


def general_bezier_curve(control_points, num_points=201):
    """
    Evaluate an (n-1)-degree Bezier curve given N control points [[x,z], ...].
    Returns u, x, z arrays.
    """
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("control_points must be an array of shape (N, 2) for 2D coordinates.")

    n = cp.shape[0]
    degree = n - 1
    if degree < 1:
        raise ValueError("Need at least 2 control points to define a Bezier curve.")

    u = np.linspace(0.0, 1.0, num_points)

    # Bernstein basis B_i(u) = C(n-1,i) u^i (1-u)^(n-1-i)
    B = np.zeros((n, len(u)))
    for i in range(n):
        coeff = binomial_coeff(degree, i)
        B[i, :] = coeff * (u ** i) * ((1 - u) ** (degree - i))

    x = np.dot(cp[:, 0], B)
    z = np.dot(cp[:, 1], B)

    return u, x, z


def bezier_point_and_tangent(control_points, u):
    """
    For a general (n-1)-degree Bezier curve with control_points [[x,z], ...],
    compute the point and tangent vector at a given scalar u in [0,1].

    Returns:
      point_2d = [x(u), z(u)]
      tangent_2d = [dx/du, dz/du]  (not normalised)
    """
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("control_points must be an array of shape (N, 2) for 2D coordinates.")

    n = cp.shape[0]
    degree = n - 1
    if degree < 1:
        raise ValueError("Need at least 2 control points to define a Bezier curve.")

    # Basis for position (degree)
    B = np.zeros(n)
    for i in range(n):
        coeff = binomial_coeff(degree, i)
        B[i] = coeff * (u ** i) * ((1 - u) ** (degree - i))

    point = np.dot(cp.T, B)  # shape (2,)

    # Basis for derivative (degree-1)
    cp_diff = degree * (cp[1:, :] - cp[:-1, :])  # shape (n-1, 2)
    Bp = np.zeros(degree)
    for j in range(degree):
        coeff = binomial_coeff(degree - 1, j)
        Bp[j] = coeff * (u ** j) * ((1 - u) ** (degree - 1 - j))
    tangent = np.dot(cp_diff.T, Bp)  # shape (2,)

    return point, tangent


def blade_length_and_thickness_from_count(num_blades):
    """
    Table 1 mapping (in metres for length, metres for thickness):
      3 → 50 m, 18 mm
      4 → 40 m, 15 mm
      5 → 30 m, 10 mm
    """
    if num_blades == 3:
        return 50.0, 0.018
    elif num_blades == 4:
        return 40.0, 0.015
    elif num_blades == 5:
        return 30.0, 0.010
    else:
        # For now, restrict to the 3 assignment variants.
        return 50.0, 0.018


def transform_tip_from_root(cp_root, scale_x, scale_z, twist_deg_per_m, blade_length):
    """
    Tip section from root (top OR bottom set):
      1) Scale in X and Z about the leading-edge position (max X)
      2) Rotate due to twist (deg per metre) about the same position
      3) Translate along Y by blade length

    cp_root: Nx2 array of [x,z] at root.
    Returns:
      cp_tip_2d: Nx2 [x_tip,z_tip]
      cp_tip_3d: Nx3 [x_tip,y_tip,z_tip]

    Coordinate system used:
      X = chord-wise
      Y = span-wise (0 at root, +L at tip)
      Z = thickness direction
    """
    cp_root = np.asarray(cp_root, dtype=float)

    # Leading edge pivot: maximum X coordinate
    x_le = np.max(cp_root[:, 0])
    z_le = 0.0  # assume LE lies on z ~ 0

    x_rel = cp_root[:, 0] - x_le
    z_rel = cp_root[:, 1] - z_le

    # Scale
    x_scaled = scale_x * x_rel
    z_scaled = scale_z * z_rel

    # Twist angle at tip
    twist_total_deg = twist_deg_per_m * blade_length
    theta = np.deg2rad(twist_total_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate about Y (blade span axis)
    x_rot = cos_t * x_scaled + sin_t * z_scaled
    z_rot = -sin_t * x_scaled + cos_t * z_scaled

    # Translate back in X,Z & along Y
    x_tip = x_rot + x_le
    z_tip = z_rot + z_le
    y_tip = np.ones_like(x_tip) * blade_length

    cp_tip_2d = np.column_stack([x_tip, z_tip])
    cp_tip_3d = np.column_stack([x_tip, y_tip, z_tip])
    return cp_tip_2d, cp_tip_3d


# =========================================================
# Excel helpers
# =========================================================

def read_template_bytes():
    """Read template from disk so we can offer it as download / default."""
    if not os.path.exists(TEMPLATE_FILENAME):
        return None
    with open(TEMPLATE_FILENAME, "rb") as f:
        return f.read()


def sort_curve_points(df_curve):
    """
    Sort rows for one curve by:
      1. PointIndex column if present
      2. Else by numeric part of Point if present (e.g. CP1,CP2,...)
      3. Else by original Excel row order
    """
    df_curve = df_curve.copy()

    if "PointIndex" in df_curve.columns:
        df_curve = df_curve.sort_values("PointIndex")
    elif "Point" in df_curve.columns:
        def extract_index(p):
            try:
                import re
                m = re.search(r"(\d+)$", str(p))
                if m:
                    return int(m.group(1))
            except Exception:
                pass
            return 999999  # unknown last

        df_curve["__idx__"] = df_curve["Point"].apply(extract_index)
        df_curve = df_curve.sort_values("__idx__").drop(columns="__idx__")
    else:
        # keep original order
        pass

    return df_curve


def load_bezier_control_points(excel_file_like):
    """
    Reads BezierControlPoints sheet from Excel and returns DataFrame, or None if missing.
    Expects at least columns: Section, Curve, X, Z for blade.
    """
    try:
        xls = pd.ExcelFile(excel_file_like)
    except Exception:
        return None

    if "BezierControlPoints" not in xls.sheet_names:
        return None

    df = pd.read_excel(xls, sheet_name="BezierControlPoints")
    return df


def load_parameters(excel_file_like):
    """
    Optional Parameters sheet:
      Columns: Name, Value
      Supported names (case-sensitive for now):
        - NumBlades
        - ScaleX
        - ScaleZ
        - TwistTotalDeg
    """
    try:
        xls = pd.ExcelFile(excel_file_like)
    except Exception:
        return {}

    if "Parameters" not in xls.sheet_names:
        return {}

    df = pd.read_excel(xls, sheet_name="Parameters")
    if "Name" not in df.columns or "Value" not in df.columns:
        return {}

    params = {}
    for _, row in df.iterrows():
        name = row["Name"]
        value = row["Value"]
        if pd.isna(name) or pd.isna(value):
            continue
        params[str(name)] = value
    return params


def load_hub_control_points(excel_file_like):
    """
    Optional HubControlPoints sheet:
      Columns: Y, Z, optionally PointIndex/Point.
    Returns numpy array of shape (N,2) [Y,Z], or None if not present.
    """
    try:
        xls = pd.ExcelFile(excel_file_like)
    except Exception:
        return None

    if "HubControlPoints" not in xls.sheet_names:
        return None

    df = pd.read_excel(xls, sheet_name="HubControlPoints")
    if not {"Y", "Z"}.issubset(df.columns):
        return None

    # Sort if PointIndex/Point given
    df_sorted = sort_curve_points(df) if ("PointIndex" in df.columns or "Point" in df.columns) else df
    Y = df_sorted["Y"].to_numpy()
    Z = df_sorted["Z"].to_numpy()

    if np.isnan(Y).any() or np.isnan(Z).any():
        return None

    cp = np.column_stack([Y, Z])
    if cp.shape[0] < 2:
        return None

    return cp


# =========================================================
# C² continuity for Curve 2 (cubic Bezier)
# =========================================================

def compute_curve2_cubic_c2(root_c1_cubic, trailing_edge=(0.0, 0.0)):
    """
    Given Curve 1 cubic control points P0,P1,P2,P3 (root_c1_cubic),
    compute Curve 2 cubic control points Q0..Q3 such that:

      - Q0 = P3  (position continuity, C⁰)
      - First derivatives at join are equal (C¹)
      - Second derivatives at join are equal (C²)
      - Q3 = trailing_edge (TE at origin for assignment)

    For cubic Bezier in 1D, the relationships at the join u=1/0 are:

      Q0 = P3
      Q1 = 2*P3 - P2
      Q2 = 4*P3 - 4*P2 + P1

    Q3 does not affect the second derivative at u=0, so we are free
    to place it at the trailing edge.

    Applied component-wise to (x,z).
    """
    P0 = root_c1_cubic[0, :]
    P1 = root_c1_cubic[1, :]
    P2 = root_c1_cubic[2, :]
    P3 = root_c1_cubic[3, :]

    Q0 = P3
    Q1 = 2.0 * P3 - P2
    Q2 = 4.0 * P3 - 4.0 * P2 + P1
    Q3 = np.array(trailing_edge, dtype=float)

    root_c2_cubic = np.vstack([Q0, Q1, Q2, Q3])
    return root_c2_cubic


# =========================================================
# Plotting helpers
# =========================================================

def apply_common_layout_tweaks(fig, title, x_range=None, z_range=None, x_label="x", z_label="z"):
    fig.update_layout(
        title=title,
        xaxis_title=f"{x_label}",
        yaxis_title=f"{z_label}",
        legend_title_text="",
        width=None,
        height=550,
        margin=dict(l=40, r=20, t=60, b=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            traceorder="normal",
        ),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(zeroline=True, zerolinewidth=2)
    fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)

    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if z_range is not None:
        fig.update_yaxes(range=z_range)

    return fig


def compute_axis_ranges(
    root_c1,
    root_c2,
    tip_c1_top_2d,
    tip_c2_top_2d,
    tip_c1_bottom_2d,
    tip_c2_bottom_2d,
):
    x_list = [root_c1[:, 0], root_c2[:, 0]]
    z_list = [root_c1[:, 1], root_c2[:, 1]]

    if tip_c1_top_2d is not None:
        x_list.append(tip_c1_top_2d[:, 0])
        z_list.append(tip_c1_top_2d[:, 1])
    if tip_c2_top_2d is not None:
        x_list.append(tip_c2_top_2d[:, 0])
        z_list.append(tip_c2_top_2d[:, 1])
    if tip_c1_bottom_2d is not None:
        x_list.append(tip_c1_bottom_2d[:, 0])
        z_list.append(tip_c1_bottom_2d[:, 1])
    if tip_c2_bottom_2d is not None:
        x_list.append(tip_c2_bottom_2d[:, 0])
        z_list.append(tip_c2_bottom_2d[:, 1])

    x_all = np.concatenate(x_list)
    z_all = np.concatenate(z_list)

    x_min, x_max = np.min(x_all), np.max(x_all)
    z_min, z_max = np.min(z_all), np.max(z_all)

    dx = x_max - x_min
    dz = z_max - z_min
    if dx == 0:
        dx = 1.0
    if dz == 0:
        dz = 1.0

    margin_x = 0.05 * dx
    margin_z = 0.05 * dz

    x_range = [x_min - margin_x, x_max + margin_x]
    z_range = [z_min - margin_z, z_max + margin_z]

    return x_range, z_range


def make_root_top_plot(root_c1, root_c2):
    u1, x1, z1_top = general_bezier_curve(root_c1)
    u2, x2, z2_top = general_bezier_curve(root_c2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=z1_top,
            mode="lines",
            name="Root Curve 1 (top)",
            line=dict(color=COLOR_ROOT_C1),
            customdata=np.column_stack([u1]),
            hovertemplate=(
                "Root Curve 1 top<br>u = %{customdata[0]:.3f}<br>"
                "x = %{x:.4f}<br>"
                "z = %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x2,
            y=z2_top,
            mode="lines",
            name="Root Curve 2 (top)",
            line=dict(color=COLOR_ROOT_C2),
            customdata=np.column_stack([u2]),
            hovertemplate=(
                "Root Curve 2 top<br>u = %{customdata[0]:.3f}<br>"
                "x = %{x:.4f}<br>"
                "z = %{y:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=root_c1[:, 0],
            y=root_c1[:, 1],
            mode="lines+markers",
            line=dict(dash="dot", color=COLOR_ROOT_C1),
            marker=dict(color=COLOR_ROOT_C1),
            name="Root Curve 1 CPs",
            hovertemplate="Root C1 CP<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=root_c2[:, 0],
            y=root_c2[:, 1],
            mode="lines+markers",
            line=dict(dash="dot", color=COLOR_ROOT_C2),
            marker=dict(color=COLOR_ROOT_C2),
            name="Root Curve 2 CPs",
            hovertemplate="Root C2 CP<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        )
    )

    fig = apply_common_layout_tweaks(
        fig,
        "Root Blade Top Profile (two cubic Bezier curves, C² at join)",
        x_range=None,
        z_range=None,
        x_label="x (chord-wise)",
        z_label="z (thickness direction)",
    )
    return fig


def make_root_and_tip_plot(
    root_c1,
    root_c2,
    tip_c1_top_2d,
    tip_c2_top_2d,
    tip_c1_bottom_2d,
    tip_c2_bottom_2d,
    show_root,
    show_tip,
    x_range,
    z_range,
):
    fig = go.Figure()

    if show_root:
        u1, x1, z1_top = general_bezier_curve(root_c1)
        z1_bottom = -z1_top

        u2, x2, z2_top = general_bezier_curve(root_c2)
        z2_bottom = -z2_top

        fig.add_trace(
            go.Scatter(
                x=x1,
                y=z1_top,
                mode="lines",
                name="Root C1 top",
                line=dict(color=COLOR_ROOT_C1),
                customdata=np.column_stack([u1]),
                hovertemplate=(
                    "Root Curve 1 top<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x2,
                y=z2_top,
                mode="lines",
                name="Root C2 top",
                line=dict(color=COLOR_ROOT_C2),
                customdata=np.column_stack([u2]),
                hovertemplate=(
                    "Root Curve 2 top<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x1,
                y=z1_bottom,
                mode="lines",
                name="Root C1 bottom",
                line=dict(color=COLOR_ROOT_C1),
                customdata=np.column_stack([u1]),
                hovertemplate=(
                    "Root Curve 1 bottom<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f} (mirrored)<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x2,
                y=z2_bottom,
                mode="lines",
                name="Root C2 bottom",
                line=dict(color=COLOR_ROOT_C2),
                customdata=np.column_stack([u2]),
                hovertemplate=(
                    "Root Curve 2 bottom<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f} (mirrored)<extra></extra>"
                ),
            )
        )

    if show_tip:
        u1_tip_top, x1_tip_top, z1_tip_top = general_bezier_curve(tip_c1_top_2d)
        u2_tip_top, x2_tip_top, z2_tip_top = general_bezier_curve(tip_c2_top_2d)
        u1_tip_bot, x1_tip_bot, z1_tip_bot = general_bezier_curve(tip_c1_bottom_2d)
        u2_tip_bot, x2_tip_bot, z2_tip_bot = general_bezier_curve(tip_c2_bottom_2d)

        fig.add_trace(
            go.Scatter(
                x=x1_tip_top,
                y=z1_tip_top,
                mode="lines",
                name="Tip C1 top",
                line=dict(dash="dash", color=COLOR_TIP_C1),
                customdata=np.column_stack([u1_tip_top]),
                hovertemplate=(
                    "Tip Curve 1 top<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x2_tip_top,
                y=z2_tip_top,
                mode="lines",
                name="Tip C2 top",
                line=dict(dash="dash", color=COLOR_TIP_C2),
                customdata=np.column_stack([u2_tip_top]),
                hovertemplate=(
                    "Tip Curve 2 top<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x1_tip_bot,
                y=z1_tip_bot,
                mode="lines",
                name="Tip C1 bottom",
                line=dict(dash="dash", color=COLOR_TIP_C1),
                customdata=np.column_stack([u1_tip_bot]),
                hovertemplate=(
                    "Tip Curve 1 bottom<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f} (mirrored)<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x2_tip_bot,
                y=z2_tip_bot,
                mode="lines",
                name="Tip C2 bottom",
                line=dict(dash="dash", color=COLOR_TIP_C2),
                customdata=np.column_stack([u2_tip_bot]),
                hovertemplate=(
                    "Tip Curve 2 bottom<br>u = %{customdata[0]:.3f}<br>"
                    "x = %{x:.4f}<br>"
                    "z = %{y:.4f} (mirrored)<extra></extra>"
                ),
            )
        )

    fig = apply_common_layout_tweaks(
        fig,
        "Blade Outer Profile (Root & Tip, mirrored about X-axis)",
        x_range=x_range,
        z_range=z_range,
        x_label="x (chord-wise)",
        z_label="z (thickness direction)",
    )
    return fig


def make_hub_plot(hub_cp_yz):
    """
    Hub profile plot in Y–Z plane.
    Input: hub_cp_yz = [[Y,Z], ...]
    """
    u, y_vals, z_vals = general_bezier_curve(hub_cp_yz)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_vals,
            y=z_vals,
            mode="lines",
            name="Hub profile",
            line=dict(color=COLOR_HUB),
            customdata=np.column_stack([u]),
            hovertemplate=(
                "Hub profile<br>u = %{customdata[0]:.3f}<br>"
                "Y = %{x:.4f}<br>"
                "Z = %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hub_cp_yz[:, 0],
            y=hub_cp_yz[:, 1],
            mode="lines+markers",
            name="Hub CPs",
            line=dict(dash="dot", color=COLOR_HUB),
            marker=dict(color=COLOR_HUB),
            hovertemplate="Hub CP<br>Y=%{x:.4f}<br>Z=%{y:.4f}<extra></extra>",
        )
    )

    fig = apply_common_layout_tweaks(
        fig,
        "Hub external profile (Bezier curve in Y–Z plane)",
        x_label="Y (radial direction)",
        z_label="Z",
    )
    return fig


def make_blade_3d_plot(root_c1, root_c2, tip_c1_top_2d, tip_c2_top_2d, blade_length_m):
    """
    Simple 3D preview:
      - root and tip curves (top and mirrored bottom),
      - a few spanwise "ribs" at selected u values.
    """
    fig = go.Figure()

    # Root curves (top/bottom) at y=0
    u1, x1_root, z1_root_top = general_bezier_curve(root_c1)
    u2, x2_root, z2_root_top = general_bezier_curve(root_c2)
    z1_root_bottom = -z1_root_top
    z2_root_bottom = -z2_root_top

    fig.add_trace(
        go.Scatter3d(
            x=x1_root,
            y=np.zeros_like(x1_root),
            z=z1_root_top,
            mode="lines",
            name="Root C1 top",
            line=dict(color=COLOR_ROOT_C1),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x2_root,
            y=np.zeros_like(x2_root),
            z=z2_root_top,
            mode="lines",
            name="Root C2 top",
            line=dict(color=COLOR_ROOT_C2),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x1_root,
            y=np.zeros_like(x1_root),
            z=z1_root_bottom,
            mode="lines",
            name="Root C1 bottom",
            line=dict(color=COLOR_ROOT_C1),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x2_root,
            y=np.zeros_like(x2_root),
            z=z2_root_bottom,
            mode="lines",
            name="Root C2 bottom",
            line=dict(color=COLOR_ROOT_C2),
        )
    )

    # Tip curves at y = blade_length_m
    u1_t, x1_tip, z1_tip_top = general_bezier_curve(tip_c1_top_2d)
    u2_t, x2_tip, z2_tip_top = general_bezier_curve(tip_c2_top_2d)
    z1_tip_bottom = -z1_tip_top
    z2_tip_bottom = -z2_tip_top

    fig.add_trace(
        go.Scatter3d(
            x=x1_tip,
            y=np.ones_like(x1_tip) * blade_length_m,
            z=z1_tip_top,
            mode="lines",
            name="Tip C1 top",
            line=dict(color=COLOR_TIP_C1),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x2_tip,
            y=np.ones_like(x2_tip) * blade_length_m,
            z=z2_tip_top,
            mode="lines",
            name="Tip C2 top",
            line=dict(color=COLOR_TIP_C2),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x1_tip,
            y=np.ones_like(x1_tip) * blade_length_m,
            z=z1_tip_bottom,
            mode="lines",
            name="Tip C1 bottom",
            line=dict(color=COLOR_TIP_C1),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x2_tip,
            y=np.ones_like(x2_tip) * blade_length_m,
            z=z2_tip_bottom,
            mode="lines",
            name="Tip C2 bottom",
            line=dict(color=COLOR_TIP_C2),
        )
    )

    # A few spanwise ribs connecting root and tip at selected u values
    for u_sel in [0.0, 0.33, 0.66, 1.0]:
        # Root points
        pt_root_c1, _ = bezier_point_and_tangent(root_c1, u_sel)
        pt_root_c2, _ = bezier_point_and_tangent(root_c2, u_sel)
        # Tip points
        pt_tip_c1, _ = bezier_point_and_tangent(tip_c1_top_2d, u_sel)
        pt_tip_c2, _ = bezier_point_and_tangent(tip_c2_top_2d, u_sel)

        # C1 top rib
        fig.add_trace(
            go.Scatter3d(
                x=[pt_root_c1[0], pt_tip_c1[0]],
                y=[0.0, blade_length_m],
                z=[pt_root_c1[1], pt_tip_c1[1]],
                mode="lines",
                showlegend=False,
                line=dict(color="#aaaaaa"),
            )
        )
        # C2 top rib
        fig.add_trace(
            go.Scatter3d(
                x=[pt_root_c2[0], pt_tip_c2[0]],
                y=[0.0, blade_length_m],
                z=[pt_root_c2[1], pt_tip_c2[1]],
                mode="lines",
                showlegend=False,
                line=dict(color="#bbbbbb"),
            )
        )

    fig.update_layout(
        title="3D preview – Blade root & tip (top and bottom surfaces)",
        scene=dict(
            xaxis_title="X (chord-wise)",
            yaxis_title="Y (span-wise)",
            zaxis_title="Z (thickness)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="top", y=0.02, xanchor="center", x=0.5),
    )
    return fig


# =========================================================
# Streamlit app
# =========================================================

def main():
    st.set_page_config(page_title="CAE4 Blade & Hub Tool", layout="wide")

    # ---------- Sidebar ----------
    with st.sidebar:
        # 1) Blade geometry inputs
        with st.expander("Blade geometry inputs", expanded=True):
            st.markdown("**Root Curve 1 – cubic control points (Figure 2)**")

            # Defaults approximated from assignment-style example:
            c1_p0_x = st.number_input("Curve 1 P0 x", value=1.0, format="%.4f")
            c1_p0_z = st.number_input("Curve 1 P0 z", value=0.0000, format="%.4f")

            c1_p1_x = st.number_input("Curve 1 P1 x", value=1.0, format="%.4f")
            c1_p1_z = st.number_input("Curve 1 P1 z", value=0.0600, format="%.4f")

            c1_p2_x = st.number_input("Curve 1 P2 x", value=0.8, format="%.4f")
            c1_p2_z = st.number_input("Curve 1 P2 z", value=0.0800, format="%.4f")

            c1_p3_x = st.number_input("Curve 1 P3 x", value=0.6, format="%.4f")
            c1_p3_z = st.number_input("Curve 1 P3 z", value=0.0750, format="%.4f")

            st.markdown("---")
            st.markdown("**Blade variant (Table 1) + tip transform**")

            num_blades_ui = st.selectbox("Number of blades (3 variants from Table 1)", [3, 4, 5], index=0)

            scale_x_ui = st.number_input(
                "Scale factor in X (example 0.8)",
                min_value=0.1,
                max_value=3.0,
                value=0.8,
                step=0.05,
            )
            scale_z_ui = st.number_input(
                "Scale factor in Z (example 0.8)",
                min_value=0.1,
                max_value=3.0,
                value=0.8,
                step=0.05,
            )

            twist_total_deg_ui = st.number_input(
                "Total twist from root to tip (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=15.0,
                step=0.5,
                format="%.2f",
            )

        # 2) Hub geometry inputs
        with st.expander("Hub geometry inputs", expanded=False):
            st.markdown("**Hub profile – cubic Bezier in Y–Z (Figure 3)**")
            st.markdown(
                "Defaults are illustrative only. Replace with control points that match Figure 3 "
                "or upload them via Excel."
            )

            hub_p0_y = st.number_input("Hub P0 Y", value=0.0, format="%.4f")
            hub_p0_z = st.number_input("Hub P0 Z", value=0.5, format="%.4f")

            hub_p1_y = st.number_input("Hub P1 Y", value=0.5, format="%.4f")
            hub_p1_z = st.number_input("Hub P1 Z", value=0.8, format="%.4f")

            hub_p2_y = st.number_input("Hub P2 Y", value=1.0, format="%.4f")
            hub_p2_z = st.number_input("Hub P2 Z", value=0.8, format="%.4f")

            hub_p3_y = st.number_input("Hub P3 Y", value=1.5, format="%.4f")
            hub_p3_z = st.number_input("Hub P3 Z", value=0.2, format="%.4f")

        # 3) Config (Excel upload etc.)
        with st.expander("Config", expanded=False):
            st.markdown("**Excel configuration (optional)**")
            st.markdown(
                """
                - If you **do not** upload a file, the tool uses the sidebar inputs.
                - If you **do** upload a file, it can override:
                  - Root Curve 1 control points (sheet `BezierControlPoints`, Section='Root', Curve=1),
                  - Hub control points (sheet `HubControlPoints`),
                  - Parameters (sheet `Parameters` with columns `Name`, `Value`), e.g.:
                    - `NumBlades` (3,4,5),
                    - `ScaleX`, `ScaleZ`,
                    - `TwistTotalDeg`.
                """
            )

            template_bytes = read_template_bytes()
            if template_bytes is not None:
                st.download_button(
                    label="📥 Download Excel template",
                    data=template_bytes,
                    file_name="blade_bezier_assignment_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_template",
                )
            else:
                st.info(
                    "Template file 'blade_bezier_assignment_template.xlsx' not found in the working directory."
                )

            uploaded_file = st.file_uploader(
                "Upload Excel configuration (optional)",
                type=["xlsx"],
                key="excel_upload",
            )

        # 4) Plot visibility toggles
        with st.expander("Plot visibility", expanded=False):
            show_root = st.checkbox("Show root curves", value=True)
            show_tip = st.checkbox("Show tip curves", value=True)

    st.title("CAE4 – Blade & Hub Parametric Tool (Assignment-aligned)")

    # ---------- Tabs ----------
    tab_overview, tab_blade, tab_3d, tab_hub, tab_cfd, tab_abaqus = st.tabs(
        ["Overview", "Blade geometry", "3D preview", "Hub geometry", "CFD domain", "Abaqus config"]
    )

    # ---------- Overview tab ----------
    with tab_overview:
        st.header("Overview")
        st.markdown(
            """
            **Blade (Figure 2)**

            - Root outer profile (top half):
              - Defined by **two cubic Bezier curves** in the X–Z plane.
              - **Curve 1**: 4 control points are the design variables (editable in the sidebar or via Excel).
              - **Curve 2**: 4 control points are computed automatically to:
                - start at the end of Curve 1,
                - end at the trailing edge at the origin (0,0),
                - satisfy **C² continuity** at the join.

            - Tip section:
              1. Same aerofoil as the root,
              2. Scaled in X and Z about the leading edge,
              3. Twisted about the same point (deg/m),
              4. Translated along +Y by the blade length (from Table 1).

            - Number of blades:
              - Limited to **3, 4, or 5**, with length and thickness from **Table 1**.

            **Hub (Figure 3)**

            - External Hub profile is defined by a **Bezier curve** in the Y–Z plane.
            - The Hub is rotated about the **x-axis through the origin**, so this profile can be revolved
              around the x-axis in CATIA.

            **CFD domain**

            - A rectangular box sized from the blade **chord** and **blade length**.
            - In CATIA, you can create this block and subtract the Hub + Blade assembly to get the fluid volume.

            **Data for CATIA / Abaqus**

            - Root & tip cubic control points (Blade curves 1 & 2),
            - Hub control points,
            - CFD domain extents,
            - Tip Curve 2 u=0.5 point and surface normal (for the 1 kN Abaqus load),
            - All exported via a **CATIA design table Excel file**, plus dedicated CSV/Script downloads for Abaqus.
            """
        )

    # ---------- Excel loading (optional) ----------
    cp_df = None
    params_from_excel = {}
    hub_cp_excel = None
    excel_loaded = False
    excel_warnings = []

    if uploaded_file is not None:
        excel_loaded = True
        excel_bytes = uploaded_file.read()

        cp_df = load_bezier_control_points(BytesIO(excel_bytes))
        if cp_df is None:
            excel_warnings.append("No or invalid `BezierControlPoints` sheet found; using sidebar CP inputs.")

        params_from_excel = load_parameters(BytesIO(excel_bytes))
        if params_from_excel:
            excel_warnings.append(
                "Using parameters from Excel `Parameters` sheet (where provided) instead of sidebar values."
            )

        hub_cp_excel = load_hub_control_points(BytesIO(excel_bytes))
        if hub_cp_excel is None:
            excel_warnings.append("No valid `HubControlPoints` sheet found; using sidebar hub CP inputs.")

    # ---------- Resolve parameters and compute geometry ONCE ----------
    # 1) Global parameters
    num_blades = int(params_from_excel.get("NumBlades", num_blades_ui))
    if num_blades not in [3, 4, 5]:
        num_blades = num_blades_ui  # fallback safety

    scale_x = float(params_from_excel.get("ScaleX", scale_x_ui))
    scale_z = float(params_from_excel.get("ScaleZ", scale_z_ui))
    twist_total_deg = float(params_from_excel.get("TwistTotalDeg", twist_total_deg_ui))

    blade_length_m, blade_thickness_m = blade_length_and_thickness_from_count(num_blades)
    twist_deg_per_m = twist_total_deg / blade_length_m if blade_length_m > 0 else 0.0

    # 2) Root Curve 1 CPs: from Excel if present, else from sidebar
    root_c1_cubic = None
    used_c1_source = "Sidebar inputs"

    if cp_df is not None:
        sub = cp_df[(cp_df.get("Section") == "Root") & (cp_df.get("Curve") == 1)]
        if not sub.empty and {"X", "Z"}.issubset(sub.columns):
            sub_sorted = sort_curve_points(sub)
            X = sub_sorted["X"].to_numpy()
            Z = sub_sorted["Z"].to_numpy()
            if not (np.isnan(X).any() or np.isnan(Z).any()) and X.shape[0] >= 4:
                root_c1_all = np.column_stack([X, Z])
                if root_c1_all.shape[0] > 4:
                    excel_warnings.append(
                        "Excel provided more than four Root / Curve 1 control points. "
                        "Using only the first four as the cubic Bezier P0..P3 (assignment requirement)."
                    )
                root_c1_cubic = root_c1_all[:4, :]
                used_c1_source = "Excel (BezierControlPoints)"
            else:
                excel_warnings.append(
                    "Excel Root / Curve 1 data incomplete or <4 points. Falling back to sidebar CPs."
                )

    if root_c1_cubic is None:
        root_c1_cubic = np.array(
            [
                [c1_p0_x, c1_p0_z],
                [c1_p1_x, c1_p1_z],
                [c1_p2_x, c1_p2_z],
                [c1_p3_x, c1_p3_z],
            ],
            dtype=float,
        )

    # 3) Root Curve 2 via C² continuity + TE at origin
    root_c2_cubic = compute_curve2_cubic_c2(root_c1_cubic, trailing_edge=(0.0, 0.0))

    # 4) Tip transform – top and bottom separately (so mirroring twists correctly)
    root_all_cps_top = np.vstack([root_c1_cubic, root_c2_cubic])
    root_all_cps_bottom = root_all_cps_top.copy()
    root_all_cps_bottom[:, 1] *= -1.0  # mirror at root, then twist

    tip_all_top_2d, tip_all_top_3d = transform_tip_from_root(
        root_all_cps_top,
        scale_x=scale_x,
        scale_z=scale_z,
        twist_deg_per_m=twist_deg_per_m,
        blade_length=blade_length_m,
    )
    tip_all_bottom_2d, _ = transform_tip_from_root(
        root_all_cps_bottom,
        scale_x=scale_x,
        scale_z=scale_z,
        twist_deg_per_m=twist_deg_per_m,
        blade_length=blade_length_m,
    )

    n1 = root_c1_cubic.shape[0]  # should be 4
    tip_c1_top_2d = tip_all_top_2d[:n1, :]
    tip_c2_top_2d = tip_all_top_2d[n1:, :]
    tip_c1_bottom_2d = tip_all_bottom_2d[:n1, :]
    tip_c2_bottom_2d = tip_all_bottom_2d[n1:, :]

    tip_c1_top_3d = tip_all_top_3d[:n1, :]
    tip_c2_top_3d = tip_all_top_3d[n1:, :]

    # 5) Axis ranges for 2D plots
    x_range, z_range = compute_axis_ranges(
        root_c1_cubic,
        root_c2_cubic,
        tip_c1_top_2d,
        tip_c2_top_2d,
        tip_c1_bottom_2d,
        tip_c2_bottom_2d,
    )

    # 6) Tip Curve 2 – u=0.5 point and normal
    u_load = 0.5
    pt2d, tan2d = bezier_point_and_tangent(tip_c2_top_2d, u_load)
    x_u, z_u = pt2d
    dxdu, dzdu = tan2d

    load_point_3d = np.array([x_u, blade_length_m, z_u], dtype=float)
    tangent_3d = np.array([dxdu, 0.0, dzdu], dtype=float)
    normal_3d = np.array([-tangent_3d[2], 0.0, tangent_3d[0]], dtype=float)
    norm_mag = np.linalg.norm(normal_3d)
    normal_unit_3d = normal_3d / norm_mag if norm_mag > 0 else np.array([0.0, 0.0, 0.0])

    # 7) Hub control points (for use in multiple tabs)
    if hub_cp_excel is not None:
        hub_cp_yz = hub_cp_excel
        hub_source_label = "Excel (HubControlPoints)"
    else:
        hub_cp_yz = np.array(
            [
                [hub_p0_y, hub_p0_z],
                [hub_p1_y, hub_p1_z],
                [hub_p2_y, hub_p2_z],
                [hub_p3_y, hub_p3_z],
            ],
            dtype=float,
        )
        hub_source_label = "Sidebar Hub CP inputs"

    # 8) Chord length and CFD domain extents
    #   - TE at x=0
    x_all_root = np.concatenate([root_c1_cubic[:, 0], root_c2_cubic[:, 0]])
    chord_length = np.max(x_all_root) - 0.0

    cfd_x_min = -CFD_X_UPSTREAM_CHORDS * chord_length
    cfd_x_max = CFD_X_DOWNSTREAM_CHORDS * chord_length
    # For simplicity, take symmetric extents in Y,Z around origin (hub centre)
    cfd_y_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_y_max = +CFD_RADIAL_CHORDS * chord_length
    cfd_z_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_z_max = +CFD_RADIAL_CHORDS * chord_length

    # 9) Manufacturing-friendly CP tables for design table export
    root_c1_df_out = pd.DataFrame(root_c1_cubic, columns=["X (m)", "Z (m)"])
    root_c1_df_out.insert(0, "Curve", "Root Curve 1")
    root_c1_df_out.insert(1, "Point index", [0, 1, 2, 3])
    root_c1_df_out.insert(2, "Y (m)", 0.0)

    root_c2_df_out = pd.DataFrame(root_c2_cubic, columns=["X (m)", "Z (m)"])
    root_c2_df_out.insert(0, "Curve", "Root Curve 2")
    root_c2_df_out.insert(1, "Point index", [0, 1, 2, 3])
    root_c2_df_out.insert(2, "Y (m)", 0.0)

    tip_c1_df_out = pd.DataFrame(tip_c1_top_3d, columns=["X (m)", "Y (m)", "Z (m)"])
    tip_c1_df_out.insert(0, "Curve", "Tip Curve 1")
    tip_c1_df_out.insert(1, "Point index", [0, 1, 2, 3])

    tip_c2_df_out = pd.DataFrame(tip_c2_top_3d, columns=["X (m)", "Y (m)", "Z (m)"])
    tip_c2_df_out.insert(0, "Curve", "Tip Curve 2")
    tip_c2_df_out.insert(1, "Point index", [0, 1, 2, 3])

    hub_df_out_for_design = pd.DataFrame(hub_cp_yz, columns=["Y (m)", "Z (m)"])
    hub_df_out_for_design.insert(0, "Point index", list(range(hub_cp_yz.shape[0])))

    # 10) CATIA design table (single configuration)
    design_params = {
        "Config": ["Default"],

        "Num_Blades": [num_blades],
        "Blade_Length": [blade_length_m],
        "Blade_Thickness": [blade_thickness_m],
        "ScaleX": [scale_x],
        "ScaleZ": [scale_z],
        "Twist_Total_Deg": [twist_total_deg],
        "Twist_Deg_Per_m": [twist_deg_per_m],

        # Root Curve 1 P0..P3
        "Root_C1_P0_X": [root_c1_cubic[0, 0]],
        "Root_C1_P0_Z": [root_c1_cubic[0, 1]],
        "Root_C1_P1_X": [root_c1_cubic[1, 0]],
        "Root_C1_P1_Z": [root_c1_cubic[1, 1]],
        "Root_C1_P2_X": [root_c1_cubic[2, 0]],
        "Root_C1_P2_Z": [root_c1_cubic[2, 1]],
        "Root_C1_P3_X": [root_c1_cubic[3, 0]],
        "Root_C1_P3_Z": [root_c1_cubic[3, 1]],

        # Root Curve 2 P0..P3
        "Root_C2_P0_X": [root_c2_cubic[0, 0]],
        "Root_C2_P0_Z": [root_c2_cubic[0, 1]],
        "Root_C2_P1_X": [root_c2_cubic[1, 0]],
        "Root_C2_P1_Z": [root_c2_cubic[1, 1]],
        "Root_C2_P2_X": [root_c2_cubic[2, 0]],
        "Root_C2_P2_Z": [root_c2_cubic[2, 1]],
        "Root_C2_P3_X": [root_c2_cubic[3, 0]],
        "Root_C2_P3_Z": [root_c2_cubic[3, 1]],

        # Tip Curve 1 P0..P3
        "Tip_C1_P0_X": [tip_c1_top_3d[0, 0]],
        "Tip_C1_P0_Y": [tip_c1_top_3d[0, 1]],
        "Tip_C1_P0_Z": [tip_c1_top_3d[0, 2]],
        "Tip_C1_P1_X": [tip_c1_top_3d[1, 0]],
        "Tip_C1_P1_Y": [tip_c1_top_3d[1, 1]],
        "Tip_C1_P1_Z": [tip_c1_top_3d[1, 2]],
        "Tip_C1_P2_X": [tip_c1_top_3d[2, 0]],
        "Tip_C1_P2_Y": [tip_c1_top_3d[2, 1]],
        "Tip_C1_P2_Z": [tip_c1_top_3d[2, 2]],
        "Tip_C1_P3_X": [tip_c1_top_3d[3, 0]],
        "Tip_C1_P3_Y": [tip_c1_top_3d[3, 1]],
        "Tip_C1_P3_Z": [tip_c1_top_3d[3, 2]],

        # Tip Curve 2 P0..P3
        "Tip_C2_P0_X": [tip_c2_top_3d[0, 0]],
        "Tip_C2_P0_Y": [tip_c2_top_3d[0, 1]],
        "Tip_C2_P0_Z": [tip_c2_top_3d[0, 2]],
        "Tip_C2_P1_X": [tip_c2_top_3d[1, 0]],
        "Tip_C2_P1_Y": [tip_c2_top_3d[1, 1]],
        "Tip_C2_P1_Z": [tip_c2_top_3d[1, 2]],
        "Tip_C2_P2_X": [tip_c2_top_3d[2, 0]],
        "Tip_C2_P2_Y": [tip_c2_top_3d[2, 1]],
        "Tip_C2_P2_Z": [tip_c2_top_3d[2, 2]],
        "Tip_C2_P3_X": [tip_c2_top_3d[3, 0]],
        "Tip_C2_P3_Y": [tip_c2_top_3d[3, 1]],
        "Tip_C2_P3_Z": [tip_c2_top_3d[3, 2]],

        # Hub CPs P0..P3 in Y–Z
        "Hub_P0_Y": [hub_cp_yz[0, 0]],
        "Hub_P0_Z": [hub_cp_yz[0, 1]],
        "Hub_P1_Y": [hub_cp_yz[1, 0]],
        "Hub_P1_Z": [hub_cp_yz[1, 1]],
        "Hub_P2_Y": [hub_cp_yz[2, 0]],
        "Hub_P2_Z": [hub_cp_yz[2, 1]],
        "Hub_P3_Y": [hub_cp_yz[3, 0]],
        "Hub_P3_Z": [hub_cp_yz[3, 1]],

        # Tip Curve 2 u=0.5 point (for load location)
        "Tip_C2_u05_X": [load_point_3d[0]],
        "Tip_C2_u05_Y": [load_point_3d[1]],
        "Tip_C2_u05_Z": [load_point_3d[2]],
        "Tip_C2_u05_NX": [normal_unit_3d[0]],
        "Tip_C2_u05_NY": [normal_unit_3d[1]],
        "Tip_C2_u05_NZ": [normal_unit_3d[2]],

        # CFD domain extents
        "CFD_Xmin": [cfd_x_min],
        "CFD_Xmax": [cfd_x_max],
        "CFD_Ymin": [cfd_y_min],
        "CFD_Ymax": [cfd_y_max],
        "CFD_Zmin": [cfd_z_min],
        "CFD_Zmax": [cfd_z_max],
        "CFD_Chord": [chord_length],
    }

    dt_df = pd.DataFrame(design_params)
    buf_dt = BytesIO()
    with pd.ExcelWriter(buf_dt, engine="xlsxwriter") as writer:
        dt_df.to_excel(writer, sheet_name="DesignTable", index=False)
        root_blade_export = pd.concat([root_c1_df_out, root_c2_df_out], ignore_index=True)
        tip_blade_export = pd.concat([tip_c1_df_out, tip_c2_df_out], ignore_index=True)
        root_blade_export.to_excel(writer, sheet_name="Blade_Root_CPs", index=False)
        tip_blade_export.to_excel(writer, sheet_name="Blade_Tip_CPs", index=False)
        hub_df_out_for_design.to_excel(writer, sheet_name="Hub_CPs", index=False)

    design_table_bytes = buf_dt.getvalue()

    # ---------- Blade geometry tab ----------
    with tab_blade:
        st.header("Blade geometry")

        if excel_loaded and excel_warnings:
            for msg in excel_warnings:
                st.info(msg)

        st.markdown(
            f"**Effective parameters used:**  "
            f"`Num blades` = {num_blades}, "
            f"`Blade length` = {blade_length_m:.2f} m, "
            f"`Thickness` = {blade_thickness_m*1000:.2f} mm, "
            f"`ScaleX` = {scale_x:.3f}, "
            f"`ScaleZ` = {scale_z:.3f}, "
            f"`Twist` = {twist_total_deg:.2f}° total ({twist_deg_per_m:.4f} °/m)."
        )
        st.markdown(f"Root Curve 1 cubic control points source: **{used_c1_source}**")

        st.subheader("Root cross-section – top surface (two cubic Bezier curves)")
        fig_top = make_root_top_plot(root_c1_cubic, root_c2_cubic)
        st.plotly_chart(fig_top, use_container_width=True)

        st.subheader("Root & tip cross-sections – top and mirrored bottom")
        fig = make_root_and_tip_plot(
            root_c1_cubic,
            root_c2_cubic,
            tip_c1_top_2d,
            tip_c2_top_2d,
            tip_c1_bottom_2d,
            tip_c2_bottom_2d,
            show_root=show_root,
            show_tip=show_tip,
            x_range=x_range,
            z_range=z_range,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tip Curve 2 – u=0.5 point and normal
        st.subheader("Tip Curve 2 – point at u = 0.5 and normal (for Abaqus load)")
        df_load = pd.DataFrame(
            {
                "Quantity": [
                    "Point X (m)",
                    "Point Y (m)",
                    "Point Z (m)",
                    "Normal X (unit)",
                    "Normal Y (unit)",
                    "Normal Z (unit)",
                ],
                "Value": [
                    load_point_3d[0],
                    load_point_3d[1],
                    load_point_3d[2],
                    normal_unit_3d[0],
                    normal_unit_3d[1],
                    normal_unit_3d[2],
                ],
            }
        )
        st.table(df_load)
        st.markdown(
            """
            - This is the **u = 0.5** location on **Tip Curve 2 (top surface)**.
            - The unit normal lies in the local cross-section plane and can be used as the direction
              for a 1 kN point load in Abaqus.
            """
        )

        st.subheader("Control points for manufacturing – Blade")

        st.markdown("**Root – cubic control points (top half, Y = 0)**")
        st.dataframe(pd.concat([root_c1_df_out, root_c2_df_out], ignore_index=True))

        st.markdown("**Tip – cubic control points (top half, Y = Blade length)**")
        st.dataframe(pd.concat([tip_c1_df_out, tip_c2_df_out], ignore_index=True))

        st.subheader("CATIA design table download (Blade + Hub + CFD + load)")
        st.download_button(
            "📥 Download CATIA design table (Excel)",
            data=design_table_bytes,
            file_name="blade_hub_cfd_design_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if cp_df is not None:
            with st.expander("Show raw `BezierControlPoints` from Excel"):
                st.dataframe(cp_df)

    # ---------- 3D preview tab ----------
    with tab_3d:
        st.header("3D preview – Blade")
        st.markdown(
            "Simple 3D visualisation of the blade root and tip sections, with a few spanwise ribs "
            "to give an impression of twist and taper."
        )
        fig_3d = make_blade_3d_plot(root_c1_cubic, root_c2_cubic, tip_c1_top_2d, tip_c2_top_2d, blade_length_m)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---------- Hub geometry tab ----------
    with tab_hub:
        st.header("Hub geometry")

        st.markdown(f"Hub profile control points source: **{hub_source_label}**")

        fig_hub = make_hub_plot(hub_cp_yz)
        st.plotly_chart(fig_hub, use_container_width=True)

        st.markdown("**Hub control points (Y–Z)**")
        hub_df_display = pd.DataFrame(hub_cp_yz, columns=["Y (m)", "Z (m)"])
        hub_df_display.insert(0, "Point index", list(range(hub_cp_yz.shape[0])))
        st.dataframe(hub_df_display)

        hub_csv = hub_df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download hub control points (CSV)",
            data=hub_csv,
            file_name="hub_control_points.csv",
            mime="text/csv",
        )

        st.markdown(
            """
            In CATIA you can:
            - Use these (Y,Z) points as a 2D sketch of the hub profile in the Y–Z plane.
            - Revolve the curve about the **x-axis through the origin** to create the hub solid.
            """
        )

    # ---------- CFD domain tab ----------
    with tab_cfd:
        st.header("CFD domain sizing")

        st.markdown(
            """
            The CFD domain is defined as a rectangular block sized from the **blade chord** and some
            dimensionless multiples. You can use these values in CATIA to create the CFD block and then
            subtract the Hub + Blade assembly to obtain the fluid volume.
            """
        )

        cfd_df = pd.DataFrame(
            {
                "Quantity": [
                    "Chord length",
                    "X-min (upstream)",
                    "X-max (downstream)",
                    "Y-min (radial)",
                    "Y-max (radial)",
                    "Z-min (vertical)",
                    "Z-max (vertical)",
                ],
                "Value [m]": [
                    chord_length,
                    cfd_x_min,
                    cfd_x_max,
                    cfd_y_min,
                    cfd_y_max,
                    cfd_z_min,
                    cfd_z_max,
                ],
            }
        )
        st.table(cfd_df)

        st.markdown(
            f"""
            - Upstream extent: {CFD_X_UPSTREAM_CHORDS:g} × chord  
            - Downstream extent: {CFD_X_DOWNSTREAM_CHORDS:g} × chord  
            - Radial/vertical extents: ±{CFD_RADIAL_CHORDS:g} × chord  
            - Hub/rotor assumed at the origin.

            In CATIA:
            1. Create a rectangular block spanning these bounds (X, Y, Z).
            2. Subtract the Hub + Blade assembly from this block (Boolean remove).
            3. Export the resulting *fluid* volume as STEP for the CFD mesher (no healing).
            """
        )

        # CSV download for CFD box
        cfd_box_df = pd.DataFrame(
            {
                "Param": ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax", "Chord"],
                "Value_m": [cfd_x_min, cfd_x_max, cfd_y_min, cfd_y_max, cfd_z_min, cfd_z_max, chord_length],
            }
        )
        cfd_csv = cfd_box_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CFD domain parameters (CSV)",
            data=cfd_csv,
            file_name="cfd_domain_parameters.csv",
            mime="text/csv",
        )

    # ---------- Abaqus config tab ----------
    with tab_abaqus:
        st.header("Abaqus configuration helpers")

        st.subheader("Load point and normal vector")
        df_load = pd.DataFrame(
            {
                "Quantity": [
                    "Point X (m)",
                    "Point Y (m)",
                    "Point Z (m)",
                    "Normal X (unit)",
                    "Normal Y (unit)",
                    "Normal Z (unit)",
                ],
                "Value": [
                    load_point_3d[0],
                    load_point_3d[1],
                    load_point_3d[2],
                    normal_unit_3d[0],
                    normal_unit_3d[1],
                    normal_unit_3d[2],
                ],
            }
        )
        st.table(df_load)

        flip_normal = st.checkbox("Flip normal direction (if Abaqus load ends up pointing outwards)", value=False)
        if flip_normal:
            normal_for_abaqus = -normal_unit_3d
        else:
            normal_for_abaqus = normal_unit_3d

        st.markdown(
            """
            The load direction for Abaqus is often specified as a **vector**. If you find that the load is
            pointing outwards instead of inwards on the blade surface, tick the checkbox above.
            """
        )

        # 1) CSV export for load point and direction
        load_data_df = pd.DataFrame(
            {
                "Quantity": [
                    "LoadPoint_X_m",
                    "LoadPoint_Y_m",
                    "LoadPoint_Z_m",
                    "LoadDir_X_unit",
                    "LoadDir_Y_unit",
                    "LoadDir_Z_unit",
                ],
                "Value": [
                    load_point_3d[0],
                    load_point_3d[1],
                    load_point_3d[2],
                    normal_for_abaqus[0],
                    normal_for_abaqus[1],
                    normal_for_abaqus[2],
                ],
            }
        )
        load_csv = load_data_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download load point & direction (CSV)",
            data=load_csv,
            file_name="abaqus_load_point_and_direction.csv",
            mime="text/csv",
        )

        # 2) Python script stub for Abaqus
        st.subheader("Abaqus Python script stub")

        script_lines = f"""# Abaqus script stub generated by CAE4 Streamlit tool
# - Imports an existing blade geometry (STEP or CATIA-exported),
# - Creates a reference point at the tip location (u=0.5 on Tip Curve 2),
# - Applies a 1 kN concentration load in the specified normal direction.

from abaqus import *
from abaqusConstants import *
import regionToolset
import assembly
import step
import mesh

MODEL_NAME = 'BladeModel'
PART_NAME  = 'BladePart'
INSTANCE_NAME = 'BladeInstance'
STEP_GEOMETRY_FILE = 'blade_geometry.step'  # TODO: replace with your geometry filename

# Load point and vector from the parametric tool:
load_point = ({load_point_3d[0]:.6f}, {load_point_3d[1]:.6f}, {load_point_3d[2]:.6f})
load_dir   = ({normal_for_abaqus[0]:.6f}, {normal_for_abaqus[1]:.6f}, {normal_for_abaqus[2]:.6f})
load_mag   = 1000.0  # N

mdb.ModelFromInputFile(name=MODEL_NAME, inputFileName=STEP_GEOMETRY_FILE)
model = mdb.models[MODEL_NAME]

# Assumes PART_NAME already exists or is deduced from the imported geometry.
part = model.parts[PART_NAME]

# Create assembly instance if needed
a = model.rootAssembly
try:
    inst = a.instances[INSTANCE_NAME]
except KeyError:
    inst = a.Instance(name=INSTANCE_NAME, part=part, dependent=ON)

# Create a reference point at the load location
rp = a.ReferencePoint(point=load_point)
rp_region = regionToolset.Region(referencePoints=(rp,))

# Create a step (simplest static step)
if 'LoadStep' not in model.steps.keys():
    model.StaticStep(name='LoadStep', previous='Initial')

# Create a concentrated load at the reference point
model.ConcentratedLoad(
    name='TipLoad',
    createStepName='LoadStep',
    region=rp_region,
    cf1=load_mag * load_dir[0],
    cf2=load_mag * load_dir[1],
    cf3=load_mag * load_dir[2],
)

# TODO:
# - Define a set for the blade root region and apply encastre BC.
# - Generate a suitable mesh (using shell or solid elements).
# - Submit the job and post-process.

print('Script setup complete. Check the geometry, sets, and part names before running.')
"""
        script_bytes = script_lines.encode("utf-8")
        st.download_button(
            "Download Abaqus Python script stub (.py)",
            data=script_bytes,
            file_name="abaqus_blade_load_stub.py",
            mime="text/x-python",
        )

        st.markdown(
            """
            This script is deliberately a **stub**:
            - You still need to adjust the part/instance names to match your imported geometry.
            - You must define the root BC region and meshing strategy.
            - But the **load location and direction** are already set from the parametric tool.
            """
        )


if __name__ == "__main__":
    main()
