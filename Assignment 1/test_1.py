import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import os
import hashlib
from typing import Dict, Any, Tuple, List, Optional

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
CFD_X_UPSTREAM_CHORDS   = 30.0
CFD_X_DOWNSTREAM_CHORDS = 120.0
CFD_RADIAL_CHORDS       = 30.0

# Sampling resolution for optional “export sampled points”
SAMPLED_POINTS_PER_CURVE = 201

# =========================================================
# Maths helpers
# =========================================================

def binomial_coeff(n, k):
    """Binomial coefficient C(n,k)."""
    from math import comb
    return comb(n, k)


def general_bezier_curve(control_points, num_points=201):
    """
    Evaluate an (n-1)-degree Bezier curve given N control points [[a,b], ...].
    Returns u, a, b arrays.
    """
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("control_points must be an array of shape (N, 2).")

    n = cp.shape[0]
    degree = n - 1
    if degree < 1:
        raise ValueError("Need at least 2 control points to define a Bezier curve.")

    u = np.linspace(0.0, 1.0, num_points)

    B = np.zeros((n, len(u)))
    for i in range(n):
        coeff = binomial_coeff(degree, i)
        B[i, :] = coeff * (u ** i) * ((1 - u) ** (degree - i))

    a = np.dot(cp[:, 0], B)
    b = np.dot(cp[:, 1], B)

    return u, a, b


def bezier_point_and_tangent(control_points, u):
    """
    For a general (n-1)-degree Bezier curve with control_points [[x,z], ...],
    compute the point and tangent vector at scalar u in [0,1].

    Returns:
      point_2d = [x(u), z(u)]
      tangent_2d = [dx/du, dz/du]  (not normalised)
    """
    cp = np.asarray(control_points, dtype=float)
    if cp.ndim != 2 or cp.shape[1] != 2:
        raise ValueError("control_points must be an array of shape (N, 2).")

    n = cp.shape[0]
    degree = n - 1
    if degree < 1:
        raise ValueError("Need at least 2 control points to define a Bezier curve.")

    # Position basis
    B = np.zeros(n)
    for i in range(n):
        coeff = binomial_coeff(degree, i)
        B[i] = coeff * (u ** i) * ((1 - u) ** (degree - i))
    point = np.dot(cp.T, B)  # (2,)

    # Derivative basis
    cp_diff = degree * (cp[1:, :] - cp[:-1, :])  # (n-1, 2)
    Bp = np.zeros(degree)
    for j in range(degree):
        coeff = binomial_coeff(degree - 1, j)
        Bp[j] = coeff * (u ** j) * ((1 - u) ** (degree - 1 - j))
    tangent = np.dot(cp_diff.T, Bp)  # (2,)

    return point, tangent


def transform_tip_from_root(cp_root, scale_x, scale_z, twist_deg_per_m, blade_length, twist_sign=+1.0):
    """
    Tip section from root:
      1) Scale in X and Z about the leading-edge position (max X)
      2) Rotate due to twist about the same position
      3) Translate along Y by blade length

    Coordinates:
      X = chord-wise
      Y = span-wise
      Z = thickness
    """
    cp_root = np.asarray(cp_root, dtype=float)

    # Leading edge pivot: maximum X coordinate
    x_le = np.max(cp_root[:, 0])
    z_le = 0.0

    x_rel = cp_root[:, 0] - x_le
    z_rel = cp_root[:, 1] - z_le

    # Scale
    x_scaled = scale_x * x_rel
    z_scaled = scale_z * z_rel

    twist_total_deg = twist_sign * twist_deg_per_m * blade_length
    theta = np.deg2rad(twist_total_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate about Y axis
    x_rot = cos_t * x_scaled + sin_t * z_scaled
    z_rot = -sin_t * x_scaled + cos_t * z_scaled

    x_tip = x_rot + x_le
    z_tip = z_rot + z_le
    y_tip = np.ones_like(x_tip) * blade_length

    cp_tip_2d = np.column_stack([x_tip, z_tip])
    cp_tip_3d = np.column_stack([x_tip, y_tip, z_tip])
    return cp_tip_2d, cp_tip_3d


# =========================================================
# Blade variant table logic (dynamic, no hard-coded restriction)
# =========================================================

def default_variant_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "NumBlades": [3, 4, 5],
            "BladeLength_m": [50.0, 40.0, 30.0],
            "Thickness_m": [0.018, 0.015, 0.010],
        }
    )

def parse_variant_table_from_excel(xls: pd.ExcelFile) -> Optional[pd.DataFrame]:
    """
    Try hard to find a blade variant table in the uploaded config.
    Accepts:
      - A sheet named 'Table1' or containing 'variant' or 'blade' and 'table'
      - Or ANY sheet containing columns that look like NumBlades + (Length/Thickness)
    """
    candidate_sheets = []
    for s in xls.sheet_names:
        name = s.lower()
        if "table1" == name or ("variant" in name) or ("blade" in name and "table" in name):
            candidate_sheets.append(s)
    # also check all sheets if needed
    candidate_sheets = candidate_sheets + [s for s in xls.sheet_names if s not in candidate_sheets]

    def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
        # Try to map common variants
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ["numblades", "num_blades", "blades", "nblades", "n_blades"]:
                col_map[c] = "NumBlades"
            if cl in ["blade_length", "bladelength", "length", "blade_length_m", "bladelength_m", "length_m"]:
                col_map[c] = "BladeLength_m"
            if cl in ["thickness", "blade_thickness", "bladethickness", "thickness_m", "blade_thickness_m"]:
                col_map[c] = "Thickness_m"
        df2 = df.rename(columns=col_map).copy()
        return df2

    for s in candidate_sheets:
        try:
            df = pd.read_excel(xls, sheet_name=s)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df2 = normalise_cols(df)
        if {"NumBlades", "BladeLength_m", "Thickness_m"}.issubset(df2.columns):
            out = df2[["NumBlades", "BladeLength_m", "Thickness_m"]].dropna()
            # coerce numeric
            try:
                out["NumBlades"] = out["NumBlades"].astype(int)
                out["BladeLength_m"] = out["BladeLength_m"].astype(float)
                out["Thickness_m"] = out["Thickness_m"].astype(float)
            except Exception:
                continue
            out = out.drop_duplicates(subset=["NumBlades"]).sort_values("NumBlades")
            if len(out) >= 1:
                return out.reset_index(drop=True)
    return None

def blade_length_and_thickness_from_table(num_blades: int, table: pd.DataFrame) -> Tuple[float, float]:
    sub = table[table["NumBlades"] == int(num_blades)]
    if sub.empty:
        # fallback: first row
        r0 = table.iloc[0]
        return float(r0["BladeLength_m"]), float(r0["Thickness_m"])
    r = sub.iloc[0]
    return float(r["BladeLength_m"]), float(r["Thickness_m"])


# =========================================================
# Excel helpers
# =========================================================

def read_template_bytes():
    if not os.path.exists(TEMPLATE_FILENAME):
        return None
    with open(TEMPLATE_FILENAME, "rb") as f:
        return f.read()


def sort_curve_points(df_curve):
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
            return 999999

        df_curve["__idx__"] = df_curve["Point"].apply(extract_index)
        df_curve = df_curve.sort_values("__idx__").drop(columns="__idx__")

    return df_curve


def load_bezier_control_points(excel_file_like):
    try:
        xls = pd.ExcelFile(excel_file_like)
    except Exception:
        return None

    if "BezierControlPoints" not in xls.sheet_names:
        return None

    return pd.read_excel(xls, sheet_name="BezierControlPoints")


def load_parameters(excel_file_like):
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
    Optional HubControlPoints sheet.

    Supports BOTH:
      - Columns: X, Z   (preferred)
      - Columns: Y, Z   (legacy fallback)

    Returns:
      (cp, used_plane_label) where cp is Nx2 and label is "X–Z" or "Y–Z"
      or (None, None)
    """
    try:
        xls = pd.ExcelFile(excel_file_like)
    except Exception:
        return None, None

    if "HubControlPoints" not in xls.sheet_names:
        return None, None

    df = pd.read_excel(xls, sheet_name="HubControlPoints")

    used = None
    if {"X", "Z"}.issubset(df.columns):
        used = "X–Z"
        df_sorted = sort_curve_points(df) if ("PointIndex" in df.columns or "Point" in df.columns) else df
        A = df_sorted["X"].to_numpy()
        B = df_sorted["Z"].to_numpy()
    elif {"Y", "Z"}.issubset(df.columns):
        used = "Y–Z"
        df_sorted = sort_curve_points(df) if ("PointIndex" in df.columns or "Point" in df.columns) else df
        A = df_sorted["Y"].to_numpy()
        B = df_sorted["Z"].to_numpy()
    else:
        return None, None

    if np.isnan(A).any() or np.isnan(B).any():
        return None, None

    cp = np.column_stack([A, B])
    if cp.shape[0] < 2:
        return None, None

    return cp, used


def file_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =========================================================
# C² continuity for Curve 2 (cubic Bezier)
# =========================================================

def compute_curve2_cubic_c2(root_c1_cubic, trailing_edge=(0.0, 0.0)):
    """
    Enforce C² at join between two cubics:
      Q0 = P3
      Q1 = 2*P3 - P2
      Q2 = 4*P3 - 4*P2 + P1
      Q3 = trailing_edge
    """
    P1 = root_c1_cubic[1, :]
    P2 = root_c1_cubic[2, :]
    P3 = root_c1_cubic[3, :]

    Q0 = P3
    Q1 = 2.0 * P3 - P2
    Q2 = 4.0 * P3 - 4.0 * P2 + P1
    Q3 = np.array(trailing_edge, dtype=float)

    return np.vstack([Q0, Q1, Q2, Q3])


def continuity_residuals_cubic(P, Q):
    """
    Return numeric residuals at the join:
      - C0: |P3 - Q0|
      - C1: |P'(1) - Q'(0)|
      - C2: |P''(1) - Q''(0)|
    For cubic:
      P'(1) = 3(P3 - P2)
      Q'(0) = 3(Q1 - Q0)
      P''(1)= 6(P3 - 2P2 + P1)
      Q''(0)= 6(Q2 - 2Q1 + Q0)
    """
    P0, P1, P2, P3 = P
    Q0, Q1, Q2, Q3 = Q

    c0 = np.linalg.norm(P3 - Q0)

    dP = 3.0 * (P3 - P2)
    dQ = 3.0 * (Q1 - Q0)
    c1 = np.linalg.norm(dP - dQ)

    ddP = 6.0 * (P3 - 2.0 * P2 + P1)
    ddQ = 6.0 * (Q2 - 2.0 * Q1 + Q0)
    c2 = np.linalg.norm(ddP - ddQ)

    return c0, c1, c2


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


def compute_axis_ranges(root_c1, root_c2, tip_c1_top_2d, tip_c2_top_2d, tip_c1_bottom_2d, tip_c2_bottom_2d):
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

    fig.add_trace(go.Scatter(
        x=x1, y=z1_top, mode="lines",
        name="Root Curve 1 (top)",
        line=dict(color=COLOR_ROOT_C1),
        customdata=np.column_stack([u1]),
        hovertemplate=(
            "Root Curve 1 top<br>u = %{customdata[0]:.3f}<br>"
            "x = %{x:.4f}<br>z = %{y:.4f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=x2, y=z2_top, mode="lines",
        name="Root Curve 2 (top)",
        line=dict(color=COLOR_ROOT_C2),
        customdata=np.column_stack([u2]),
        hovertemplate=(
            "Root Curve 2 top<br>u = %{customdata[0]:.3f}<br>"
            "x = %{x:.4f}<br>z = %{y:.4f}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=root_c1[:, 0], y=root_c1[:, 1], mode="lines+markers",
        line=dict(dash="dot", color=COLOR_ROOT_C1),
        marker=dict(color=COLOR_ROOT_C1),
        name="Root Curve 1 CPs",
        hovertemplate="Root C1 CP<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=root_c2[:, 0], y=root_c2[:, 1], mode="lines+markers",
        line=dict(dash="dot", color=COLOR_ROOT_C2),
        marker=dict(color=COLOR_ROOT_C2),
        name="Root Curve 2 CPs",
        hovertemplate="Root C2 CP<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
    ))

    return apply_common_layout_tweaks(
        fig,
        "Root Blade Top Profile (two cubic Bezier curves, C² at join)",
        x_label="x (chord-wise)",
        z_label="z (thickness direction)",
    )


def make_root_and_tip_plot(
    root_c1, root_c2,
    tip_c1_top_2d, tip_c2_top_2d,
    tip_c1_bottom_2d, tip_c2_bottom_2d,
    show_root, show_tip,
    x_range, z_range,
):
    fig = go.Figure()

    if show_root:
        u1, x1, z1_top = general_bezier_curve(root_c1)
        u2, x2, z2_top = general_bezier_curve(root_c2)

        z1_bottom = -z1_top
        z2_bottom = -z2_top

        fig.add_trace(go.Scatter(
            x=x1, y=z1_top, mode="lines", name="Root C1 top",
            line=dict(color=COLOR_ROOT_C1),
            customdata=np.column_stack([u1]),
            hovertemplate="Root C1 top<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x2, y=z2_top, mode="lines", name="Root C2 top",
            line=dict(color=COLOR_ROOT_C2),
            customdata=np.column_stack([u2]),
            hovertemplate="Root C2 top<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x1, y=z1_bottom, mode="lines", name="Root C1 bottom",
            line=dict(color=COLOR_ROOT_C1),
            customdata=np.column_stack([u1]),
            hovertemplate="Root C1 bottom<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f} (mirrored)<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x2, y=z2_bottom, mode="lines", name="Root C2 bottom",
            line=dict(color=COLOR_ROOT_C2),
            customdata=np.column_stack([u2]),
            hovertemplate="Root C2 bottom<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f} (mirrored)<extra></extra>",
        ))

    if show_tip:
        u1t, x1t, z1t = general_bezier_curve(tip_c1_top_2d)
        u2t, x2t, z2t = general_bezier_curve(tip_c2_top_2d)
        u1b, x1b, z1b = general_bezier_curve(tip_c1_bottom_2d)
        u2b, x2b, z2b = general_bezier_curve(tip_c2_bottom_2d)

        fig.add_trace(go.Scatter(
            x=x1t, y=z1t, mode="lines", name="Tip C1 top",
            line=dict(dash="dash", color=COLOR_TIP_C1),
            customdata=np.column_stack([u1t]),
            hovertemplate="Tip C1 top<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x2t, y=z2t, mode="lines", name="Tip C2 top",
            line=dict(dash="dash", color=COLOR_TIP_C2),
            customdata=np.column_stack([u2t]),
            hovertemplate="Tip C2 top<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x1b, y=z1b, mode="lines", name="Tip C1 bottom",
            line=dict(dash="dash", color=COLOR_TIP_C1),
            customdata=np.column_stack([u1b]),
            hovertemplate="Tip C1 bottom<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f} (mirrored)<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x2b, y=z2b, mode="lines", name="Tip C2 bottom",
            line=dict(dash="dash", color=COLOR_TIP_C2),
            customdata=np.column_stack([u2b]),
            hovertemplate="Tip C2 bottom<br>u=%{customdata[0]:.3f}<br>x=%{x:.4f}<br>z=%{y:.4f} (mirrored)<extra></extra>",
        ))

    return apply_common_layout_tweaks(
        fig,
        "Blade Outer Profile (Root & Tip, mirrored about X-axis)",
        x_range=x_range,
        z_range=z_range,
        x_label="x (chord-wise)",
        z_label="z (thickness direction)",
    )


def make_hub_plot(hub_cp_az, hub_plane_label="X–Z"):
    """
    Hub profile plot in the chosen plane:
      - if hub_plane_label = "X–Z", hub_cp_az is [[X,Z], ...]
      - if hub_plane_label = "Y–Z", hub_cp_az is [[Y,Z], ...]
    """
    u, a_vals, z_vals = general_bezier_curve(hub_cp_az)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=a_vals, y=z_vals, mode="lines",
        name="Hub profile",
        line=dict(color=COLOR_HUB),
        customdata=np.column_stack([u]),
        hovertemplate=(
            f"Hub profile ({hub_plane_label})<br>u=%{{customdata[0]:.3f}}<br>"
            f"{hub_plane_label.split('–')[0]}=%{{x:.4f}}<br>Z=%{{y:.4f}}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=hub_cp_az[:, 0], y=hub_cp_az[:, 1],
        mode="lines+markers",
        name="Hub CPs",
        line=dict(dash="dot", color=COLOR_HUB),
        marker=dict(color=COLOR_HUB),
        hovertemplate=(
            f"Hub CP<br>{hub_plane_label.split('–')[0]}=%{{x:.4f}}<br>Z=%{{y:.4f}}<extra></extra>"
        ),
    ))

    xlab = f"{hub_plane_label.split('–')[0]} (hub plane)"
    return apply_common_layout_tweaks(fig, f"Hub external profile (Bezier curve in {hub_plane_label})",
                                     x_label=xlab, z_label="Z")


def make_blade_3d_plot(root_c1, root_c2, tip_c1_top_2d, tip_c2_top_2d, blade_length_m):
    fig = go.Figure()

    # Root (y=0)
    _, x1r, z1r_top = general_bezier_curve(root_c1)
    _, x2r, z2r_top = general_bezier_curve(root_c2)
    fig.add_trace(go.Scatter3d(x=x1r, y=np.zeros_like(x1r), z=z1r_top, mode="lines", name="Root C1 top", line=dict(color=COLOR_ROOT_C1)))
    fig.add_trace(go.Scatter3d(x=x2r, y=np.zeros_like(x2r), z=z2r_top, mode="lines", name="Root C2 top", line=dict(color=COLOR_ROOT_C2)))
    fig.add_trace(go.Scatter3d(x=x1r, y=np.zeros_like(x1r), z=-z1r_top, mode="lines", name="Root C1 bottom", line=dict(color=COLOR_ROOT_C1)))
    fig.add_trace(go.Scatter3d(x=x2r, y=np.zeros_like(x2r), z=-z2r_top, mode="lines", name="Root C2 bottom", line=dict(color=COLOR_ROOT_C2)))

    # Tip (y=L)
    _, x1t, z1t_top = general_bezier_curve(tip_c1_top_2d)
    _, x2t, z2t_top = general_bezier_curve(tip_c2_top_2d)
    fig.add_trace(go.Scatter3d(x=x1t, y=np.ones_like(x1t) * blade_length_m, z=z1t_top, mode="lines", name="Tip C1 top", line=dict(color=COLOR_TIP_C1)))
    fig.add_trace(go.Scatter3d(x=x2t, y=np.ones_like(x2t) * blade_length_m, z=z2t_top, mode="lines", name="Tip C2 top", line=dict(color=COLOR_TIP_C2)))
    fig.add_trace(go.Scatter3d(x=x1t, y=np.ones_like(x1t) * blade_length_m, z=-z1t_top, mode="lines", name="Tip C1 bottom", line=dict(color=COLOR_TIP_C1)))
    fig.add_trace(go.Scatter3d(x=x2t, y=np.ones_like(x2t) * blade_length_m, z=-z2t_top, mode="lines", name="Tip C2 bottom", line=dict(color=COLOR_TIP_C2)))

    # Ribs
    for u_sel in [0.0, 0.33, 0.66, 1.0]:
        pt_root_c1, _ = bezier_point_and_tangent(root_c1, u_sel)
        pt_root_c2, _ = bezier_point_and_tangent(root_c2, u_sel)
        pt_tip_c1, _ = bezier_point_and_tangent(tip_c1_top_2d, u_sel)
        pt_tip_c2, _ = bezier_point_and_tangent(tip_c2_top_2d, u_sel)

        fig.add_trace(go.Scatter3d(
            x=[pt_root_c1[0], pt_tip_c1[0]],
            y=[0.0, blade_length_m],
            z=[pt_root_c1[1], pt_tip_c1[1]],
            mode="lines", showlegend=False, line=dict(color="#aaaaaa"),
        ))
        fig.add_trace(go.Scatter3d(
            x=[pt_root_c2[0], pt_tip_c2[0]],
            y=[0.0, blade_length_m],
            z=[pt_root_c2[1], pt_tip_c2[1]],
            mode="lines", showlegend=False, line=dict(color="#bbbbbb"),
        ))

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
# Export helpers (sampled points)
# =========================================================

def sampled_blade_points_csv(root_c1, root_c2, tip_c1_top_2d, tip_c2_top_2d, tip_c1_bot_2d, tip_c2_bot_2d, blade_length_m):
    """
    Produce a CSV of sampled points for:
      - Root top/bottom (Y=0)
      - Tip top/bottom (Y=blade_length_m)
    """
    rows = []

    def add_curve(section, surface, curve_name, y_val, cp_2d):
        u, x, z = general_bezier_curve(cp_2d, num_points=SAMPLED_POINTS_PER_CURVE)
        for ui, xi, zi in zip(u, x, z):
            rows.append({
                "Section": section,
                "Surface": surface,
                "Curve": curve_name,
                "u": float(ui),
                "X": float(xi),
                "Y": float(y_val),
                "Z": float(zi),
            })

    # Root
    add_curve("Root", "Top", "C1", 0.0, root_c1)
    add_curve("Root", "Top", "C2", 0.0, root_c2)
    add_curve("Root", "Bottom", "C1", 0.0, root_c1 * np.array([1.0, -1.0]))
    add_curve("Root", "Bottom", "C2", 0.0, root_c2 * np.array([1.0, -1.0]))

    # Tip
    add_curve("Tip", "Top", "C1", blade_length_m, tip_c1_top_2d)
    add_curve("Tip", "Top", "C2", blade_length_m, tip_c2_top_2d)
    add_curve("Tip", "Bottom", "C1", blade_length_m, tip_c1_bot_2d)
    add_curve("Tip", "Bottom", "C2", blade_length_m, tip_c2_bot_2d)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# Config behaviour: apply uploaded config ONCE per file change
# =========================================================

def ensure_state_defaults():
    """
    Create baseline defaults only once per session.
    These are used on first run OR before any config upload.
    """
    if "defaults_initialised" in st.session_state:
        return

    st.session_state.defaults_initialised = True

    # Blade defaults
    st.session_state.blade_options = [3, 4, 5]
    st.session_state.num_blades = 3
    st.session_state.scale_x = 0.8
    st.session_state.scale_z = 0.8
    st.session_state.twist_total_deg = 15.0
    st.session_state.twist_sign = +1.0

    st.session_state.c1_p0_x = 1.0
    st.session_state.c1_p0_z = 0.0
    st.session_state.c1_p1_x = 1.0
    st.session_state.c1_p1_z = 0.0600
    st.session_state.c1_p2_x = 0.8
    st.session_state.c1_p2_z = 0.0800
    st.session_state.c1_p3_x = 0.6
    st.session_state.c1_p3_z = 0.0750

    # Hub defaults (X–Z)
    st.session_state.hub_p0_x = -0.4
    st.session_state.hub_p0_z = 1.0
    st.session_state.hub_p1_x = 0.3
    st.session_state.hub_p1_z = 1.0
    st.session_state.hub_p2_x = 13.0
    st.session_state.hub_p2_z = 0.8
    st.session_state.hub_p3_x = 2.0
    st.session_state.hub_p3_z = 0.0

    # Plot visibility
    st.session_state.show_root = True
    st.session_state.show_tip = True

    # Abaqus defaults
    st.session_state.abaqus_load_mag = 1000.0
    st.session_state.abaqus_u = 0.5
    st.session_state.abaqus_flip_normal = False
    st.session_state.abaqus_target = "Tip_C2_Top"  # requested default

    # Config tracking
    st.session_state.last_config_hash = None


def apply_config_once_if_new(uploaded_bytes: bytes) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    If a config is uploaded and is new, overwrite session_state defaults + dropdown options.
    Do NOT keep re-applying on every rerun (prevents reverting user edits).
    """
    warnings = []
    applied = False
    variant_table = default_variant_table()

    if uploaded_bytes is None:
        return applied, warnings, variant_table

    cfg_hash = file_hash_bytes(uploaded_bytes)
    if st.session_state.last_config_hash == cfg_hash:
        # Same file as last time: do nothing
        return applied, warnings, variant_table

    # New config file: apply it ONCE, and record hash
    st.session_state.last_config_hash = cfg_hash
    applied = True

    try:
        xls = pd.ExcelFile(BytesIO(uploaded_bytes))
    except Exception:
        warnings.append("Config upload could not be read as an Excel file. Using current sidebar values.")
        return applied, warnings, variant_table

    # Variant table (this is what drives dynamic blade dropdown options)
    vt = parse_variant_table_from_excel(xls)
    if vt is not None:
        variant_table = vt
        new_opts = list(map(int, variant_table["NumBlades"].tolist()))
        if len(new_opts) >= 1:
            st.session_state.blade_options = new_opts
            # If current selection not in new options, set to first option
            if int(st.session_state.num_blades) not in new_opts:
                st.session_state.num_blades = int(new_opts[0])
    else:
        warnings.append("No blade variant table found in config; using default blade options (3,4,5).")

    # Parameters sheet (optional overrides)
    params = load_parameters(BytesIO(uploaded_bytes))
    if params:
        # Common parameters: update defaults visible in UI
        def try_set_float(key_state: str, param_name: str):
            if param_name in params:
                try:
                    st.session_state[key_state] = float(params[param_name])
                except Exception:
                    warnings.append(f"Parameter '{param_name}' was not a valid number; ignored.")

        def try_set_int(key_state: str, param_name: str):
            if param_name in params:
                try:
                    st.session_state[key_state] = int(float(params[param_name]))
                except Exception:
                    warnings.append(f"Parameter '{param_name}' was not a valid integer; ignored.")

        try_set_int("num_blades", "NumBlades")
        # Ensure selection valid vs options (if options exist)
        if "blade_options" in st.session_state and st.session_state.blade_options:
            if int(st.session_state.num_blades) not in st.session_state.blade_options:
                st.session_state.num_blades = int(st.session_state.blade_options[0])

        try_set_float("scale_x", "ScaleX")
        try_set_float("scale_z", "ScaleZ")
        try_set_float("twist_total_deg", "TwistTotalDeg")

        if "TwistSign" in params:
            try:
                st.session_state.twist_sign = +1.0 if float(params["TwistSign"]) >= 0 else -1.0
            except Exception:
                warnings.append("Parameter 'TwistSign' invalid; ignored.")

    # Root control points from BezierControlPoints sheet (optional)
    cp_df = load_bezier_control_points(BytesIO(uploaded_bytes))
    if cp_df is not None:
        sub = cp_df[(cp_df.get("Section") == "Root") & (cp_df.get("Curve") == 1)]
        if not sub.empty and {"X", "Z"}.issubset(sub.columns):
            sub_sorted = sort_curve_points(sub)
            X = sub_sorted["X"].to_numpy()
            Z = sub_sorted["Z"].to_numpy()
            if not (np.isnan(X).any() or np.isnan(Z).any()) and X.shape[0] >= 4:
                # Apply first four
                st.session_state.c1_p0_x = float(X[0]); st.session_state.c1_p0_z = float(Z[0])
                st.session_state.c1_p1_x = float(X[1]); st.session_state.c1_p1_z = float(Z[1])
                st.session_state.c1_p2_x = float(X[2]); st.session_state.c1_p2_z = float(Z[2])
                st.session_state.c1_p3_x = float(X[3]); st.session_state.c1_p3_z = float(Z[3])
            else:
                warnings.append("BezierControlPoints Root/Curve1 found but invalid; ignored.")

    # Hub CPs from HubControlPoints (optional) -> apply first 4 if present
    hub_cp, hub_plane = load_hub_control_points(BytesIO(uploaded_bytes))
    if hub_cp is not None and hub_cp.shape[0] >= 4:
        # If config gives Y–Z legacy, we still store into hub_p*_x for CATIA design naming (your request)
        st.session_state.hub_p0_x = float(hub_cp[0, 0]); st.session_state.hub_p0_z = float(hub_cp[0, 1])
        st.session_state.hub_p1_x = float(hub_cp[1, 0]); st.session_state.hub_p1_z = float(hub_cp[1, 1])
        st.session_state.hub_p2_x = float(hub_cp[2, 0]); st.session_state.hub_p2_z = float(hub_cp[2, 1])
        st.session_state.hub_p3_x = float(hub_cp[3, 0]); st.session_state.hub_p3_z = float(hub_cp[3, 1])
    elif hub_cp is None:
        warnings.append("No valid HubControlPoints found in config; using current hub defaults.")

    # Abaqus defaults from Parameters (optional)
    if params:
        if "AbaqusLoadMag" in params:
            try:
                st.session_state.abaqus_load_mag = float(params["AbaqusLoadMag"])
            except Exception:
                warnings.append("AbaqusLoadMag invalid; ignored.")
        if "AbaqusU" in params:
            try:
                st.session_state.abaqus_u = float(params["AbaqusU"])
            except Exception:
                warnings.append("AbaqusU invalid; ignored.")
        if "AbaqusTarget" in params:
            st.session_state.abaqus_target = str(params["AbaqusTarget"])

    return applied, warnings, variant_table


# =========================================================
# Streamlit app
# =========================================================

def main():
    st.set_page_config(page_title="CAE4 Blade & Hub Tool", layout="wide")

    ensure_state_defaults()

    # ---------- Sidebar (upload must happen early so it can update session_state BEFORE widgets) ----------
    with st.sidebar:
        with st.expander("Config", expanded=False):
            st.markdown("**Excel configuration (optional)**")
            uploaded_file = st.file_uploader(
                "Upload Excel configuration (optional)",
                type=["xlsx"],
                key="excel_upload",
            )
        uploaded_bytes = uploaded_file.read() if uploaded_file is not None else None

    # Apply config once per new file (this is what prevents reverting-on-edit)
    config_applied, config_warnings, variant_table = apply_config_once_if_new(uploaded_bytes)

    # ---------- Sidebar (now build all widgets, using session_state keys) ----------
    with st.sidebar:
        with st.expander("Blade geometry inputs", expanded=True):
            st.markdown("**Root Curve 1 – Control Points**")

            st.number_input("Curve 1 P0 x", key="c1_p0_x", format="%.4f")
            st.number_input("Curve 1 P0 z", key="c1_p0_z", format="%.4f")

            st.number_input("Curve 1 P1 x", key="c1_p1_x", format="%.4f")
            st.number_input("Curve 1 P1 z", key="c1_p1_z", format="%.4f")

            st.number_input("Curve 1 P2 x", key="c1_p2_x", format="%.4f")
            st.number_input("Curve 1 P2 z", key="c1_p2_z", format="%.4f")

            st.number_input("Curve 1 P3 x", key="c1_p3_x", format="%.4f")
            st.number_input("Curve 1 P3 z", key="c1_p3_z", format="%.4f")

            st.markdown("---")
            st.markdown("**Blade variant (Table 1) + tip transform**")

            # Dynamic options (updated by config)
            blade_opts = st.session_state.get("blade_options", [3, 4, 5])
            if not blade_opts:
                blade_opts = [3, 4, 5]
                st.session_state.blade_options = blade_opts

            # Ensure current selection valid
            if int(st.session_state.num_blades) not in blade_opts:
                st.session_state.num_blades = int(blade_opts[0])

            st.selectbox(
                "Number of blades (from variant table)",
                options=blade_opts,
                key="num_blades",
            )

            st.number_input("Scale factor in X (example 0.8)", min_value=0.1, max_value=3.0, step=0.05, key="scale_x")
            st.number_input("Scale factor in Z (example 0.8)", min_value=0.1, max_value=3.0, step=0.05, key="scale_z")

            st.number_input(
                "Total twist from root to tip (degrees)",
                min_value=0.0,
                max_value=90.0,
                step=0.5,
                format="%.2f",
                key="twist_total_deg",
            )
            twist_direction = st.selectbox(
                "Twist direction",
                ["Positive", "Negative"],
                index=0 if st.session_state.twist_sign >= 0 else 1,
                key="twist_direction_ui",
            )
            st.session_state.twist_sign = +1.0 if twist_direction == "Positive" else -1.0

        with st.expander("Hub geometry inputs", expanded=False):
            st.markdown("**Hub profile – Control Points (X–Z by default)**")
            st.number_input("Hub P0 X", key="hub_p0_x", format="%.4f")
            st.number_input("Hub P0 Z", key="hub_p0_z", format="%.4f")

            st.number_input("Hub P1 X", key="hub_p1_x", format="%.4f")
            st.number_input("Hub P1 Z", key="hub_p1_z", format="%.4f")

            st.number_input("Hub P2 X", key="hub_p2_x", format="%.4f")
            st.number_input("Hub P2 Z", key="hub_p2_z", format="%.4f")

            st.number_input("Hub P3 X", key="hub_p3_x", format="%.4f")
            st.number_input("Hub P3 Z", key="hub_p3_z", format="%.4f")

        with st.expander("Plot visibility", expanded=False):
            st.checkbox("Show root curves", key="show_root")
            st.checkbox("Show tip curves", key="show_tip")

        # Abaqus controls in sidebar (as requested)
        with st.expander("Abaqus", expanded=False):
            st.number_input("Load magnitude (N)", min_value=0.0, step=50.0, key="abaqus_load_mag")
            st.slider("Force location u (0 to 1)", min_value=0.0, max_value=1.0, step=0.01, key="abaqus_u")

            # Combined dropdown: Section_Curve_Surface
            targets = [
                "Root_C1_Top", "Root_C2_Top", "Tip_C1_Top", "Tip_C2_Top",
                "Root_C1_Bottom", "Root_C2_Bottom", "Tip_C1_Bottom", "Tip_C2_Bottom",
            ]
            # Ensure default exists
            if st.session_state.abaqus_target not in targets:
                st.session_state.abaqus_target = "Tip_C2_Top"

            st.selectbox(
                "Apply load to",
                options=targets,
                key="abaqus_target",
            )
            st.checkbox("Flip normal direction", key="abaqus_flip_normal")

        # Downloads expander (CATIA + config template + Abaqus script)
        # We create the files after computing geometry, but place the buttons here later using placeholders.
        downloads_placeholder = st.empty()

    st.title("CAE4 – Blade & Hub Parametric Tool")

    # ---------- Tabs ----------
    tab_overview, tab_blade, tab_3d, tab_hub, tab_cfd, tab_abaqus = st.tabs(
        ["Overview", "Blade geometry", "3D preview", "Hub geometry", "CFD domain", "Abaqus config"]
    )

    # ---------- Overview tab ----------
    with tab_overview:
        st.header("Overview")
        if config_applied:
            st.success("Config applied (as new defaults). You can now tweak values in the sidebar without them reverting.")
        if config_warnings:
            for w in config_warnings:
                st.info(w)

        st.markdown(
            """
            **Blade**
            - Root profile: 2 cubic Bezier curves in X–Z, Curve 2 computed for C² at join and trailing edge at origin.
            - Tip: scale + twist + translate along Y.

            **Hub**
            - Hub profile uses control points in X–Z (preferred) or Y–Z (legacy).
            - Design table exports hub points as X/Z fields (CATIA-side naming).

            **Abaqus**
            - Choose a target curve/surface, pick u location, and export a Python stub.
            """
        )

    # =========================================================
    # Compute geometry from current (editable) session_state
    # =========================================================

    # Variant table drives length/thickness
    num_blades = int(st.session_state.num_blades)
    blade_length_m, blade_thickness_m = blade_length_and_thickness_from_table(num_blades, variant_table)

    scale_x = float(st.session_state.scale_x)
    scale_z = float(st.session_state.scale_z)
    twist_total_deg = float(st.session_state.twist_total_deg)
    twist_sign = float(st.session_state.twist_sign)

    twist_deg_per_m = twist_total_deg / blade_length_m if blade_length_m > 0 else 0.0

    # Root Curve 1 CPs
    root_c1_cubic = np.array(
        [
            [st.session_state.c1_p0_x, st.session_state.c1_p0_z],
            [st.session_state.c1_p1_x, st.session_state.c1_p1_z],
            [st.session_state.c1_p2_x, st.session_state.c1_p2_z],
            [st.session_state.c1_p3_x, st.session_state.c1_p3_z],
        ],
        dtype=float,
    )

    # Root Curve 2 via C² continuity + TE at origin
    root_c2_cubic = compute_curve2_cubic_c2(root_c1_cubic, trailing_edge=(0.0, 0.0))
    c0_res, c1_res, c2_res = continuity_residuals_cubic(root_c1_cubic, root_c2_cubic)

    # Tip transform – top and bottom separately
    root_all_cps_top = np.vstack([root_c1_cubic, root_c2_cubic])
    root_all_cps_bottom = root_all_cps_top.copy()
    root_all_cps_bottom[:, 1] *= -1.0

    tip_all_top_2d, tip_all_top_3d = transform_tip_from_root(
        root_all_cps_top, scale_x=scale_x, scale_z=scale_z,
        twist_deg_per_m=twist_deg_per_m, blade_length=blade_length_m,
        twist_sign=twist_sign,
    )
    tip_all_bottom_2d, tip_all_bottom_3d = transform_tip_from_root(
        root_all_cps_bottom, scale_x=scale_x, scale_z=scale_z,
        twist_deg_per_m=twist_deg_per_m, blade_length=blade_length_m,
        twist_sign=twist_sign,
    )

    n1 = 4
    tip_c1_top_2d = tip_all_top_2d[:n1, :]
    tip_c2_top_2d = tip_all_top_2d[n1:, :]
    tip_c1_bottom_2d = tip_all_bottom_2d[:n1, :]
    tip_c2_bottom_2d = tip_all_bottom_2d[n1:, :]

    tip_c1_top_3d = tip_all_top_3d[:n1, :]
    tip_c2_top_3d = tip_all_top_3d[n1:, :]

    # Axis ranges
    x_range, z_range = compute_axis_ranges(
        root_c1_cubic, root_c2_cubic,
        tip_c1_top_2d, tip_c2_top_2d,
        tip_c1_bottom_2d, tip_c2_bottom_2d,
    )

    # Hub CPs (sidebar is X–Z)
    hub_cp_az = np.array(
        [
            [st.session_state.hub_p0_x, st.session_state.hub_p0_z],
            [st.session_state.hub_p1_x, st.session_state.hub_p1_z],
            [st.session_state.hub_p2_x, st.session_state.hub_p2_z],
            [st.session_state.hub_p3_x, st.session_state.hub_p3_z],
        ],
        dtype=float,
    )
    hub_plane_label = "X–Z"
    hub_source_label = "Sidebar Hub CP inputs (X–Z)"

    # Chord length and CFD domain extents (root)
    x_all_root = np.concatenate([root_c1_cubic[:, 0], root_c2_cubic[:, 0]])
    chord_length = float(np.max(x_all_root) - 0.0)

    cfd_x_min = -CFD_X_UPSTREAM_CHORDS * chord_length
    cfd_x_max = CFD_X_DOWNSTREAM_CHORDS * chord_length
    cfd_y_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_y_max = +CFD_RADIAL_CHORDS * chord_length
    cfd_z_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_z_max = +CFD_RADIAL_CHORDS * chord_length

    # Abaqus: build the target curve arrays (2D x-z, plus the span y)
    curves_2d = {
        "Root_C1_Top": root_c1_cubic,
        "Root_C2_Top": root_c2_cubic,
        "Tip_C1_Top": tip_c1_top_2d,
        "Tip_C2_Top": tip_c2_top_2d,
        "Root_C1_Bottom": root_c1_cubic * np.array([1.0, -1.0]),
        "Root_C2_Bottom": root_c2_cubic * np.array([1.0, -1.0]),
        "Tip_C1_Bottom": tip_c1_bottom_2d,
        "Tip_C2_Bottom": tip_c2_bottom_2d,
    }
    curve_to_y = {
        "Root_C1_Top": 0.0, "Root_C2_Top": 0.0, "Root_C1_Bottom": 0.0, "Root_C2_Bottom": 0.0,
        "Tip_C1_Top": blade_length_m, "Tip_C2_Top": blade_length_m, "Tip_C1_Bottom": blade_length_m, "Tip_C2_Bottom": blade_length_m,
    }

    abaqus_target = st.session_state.abaqus_target
    abaqus_u = float(st.session_state.abaqus_u)
    load_mag = float(st.session_state.abaqus_load_mag)

    target_curve_2d = curves_2d.get(abaqus_target, tip_c2_top_2d)
    y_for_target = float(curve_to_y.get(abaqus_target, blade_length_m))

    pt2d, tan2d = bezier_point_and_tangent(target_curve_2d, abaqus_u)
    x_u, z_u = float(pt2d[0]), float(pt2d[1])
    dxdu, dzdu = float(tan2d[0]), float(tan2d[1])

    load_point_3d = np.array([x_u, y_for_target, z_u], dtype=float)

    tangent_3d = np.array([dxdu, 0.0, dzdu], dtype=float)
    normal_3d = np.array([-tangent_3d[2], 0.0, tangent_3d[0]], dtype=float)
    norm_mag = float(np.linalg.norm(normal_3d))
    normal_unit_3d = normal_3d / norm_mag if norm_mag > 0 else np.array([0.0, 0.0, 0.0])

    if bool(st.session_state.abaqus_flip_normal):
        normal_unit_3d = -normal_unit_3d

    # =========================================================
    # Output tables for exports
    # =========================================================

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

    hub_df_out_for_design = pd.DataFrame(hub_cp_az, columns=["X (m)", "Z (m)"])
    hub_df_out_for_design.insert(0, "Point index", list(range(hub_cp_az.shape[0])))

    # =========================================================
    # CATIA design table (single configuration)
    # =========================================================

    design_params = {
        "Config": ["Default"],

        "Num_Blades": [num_blades],
        "Blade_Length": [blade_length_m],
        "Blade_Thickness": [blade_thickness_m],
        "ScaleX": [scale_x],
        "ScaleZ": [scale_z],
        "Twist_Total_Deg": [twist_total_deg],
        "Twist_Deg_Per_m": [twist_deg_per_m],
        "Twist_Sign": [twist_sign],

        # Root Curve 1
        "Root_C1_P0_X": [root_c1_cubic[0, 0]], "Root_C1_P0_Z": [root_c1_cubic[0, 1]],
        "Root_C1_P1_X": [root_c1_cubic[1, 0]], "Root_C1_P1_Z": [root_c1_cubic[1, 1]],
        "Root_C1_P2_X": [root_c1_cubic[2, 0]], "Root_C1_P2_Z": [root_c1_cubic[2, 1]],
        "Root_C1_P3_X": [root_c1_cubic[3, 0]], "Root_C1_P3_Z": [root_c1_cubic[3, 1]],

        # Root Curve 2
        "Root_C2_P0_X": [root_c2_cubic[0, 0]], "Root_C2_P0_Z": [root_c2_cubic[0, 1]],
        "Root_C2_P1_X": [root_c2_cubic[1, 0]], "Root_C2_P1_Z": [root_c2_cubic[1, 1]],
        "Root_C2_P2_X": [root_c2_cubic[2, 0]], "Root_C2_P2_Z": [root_c2_cubic[2, 1]],
        "Root_C2_P3_X": [root_c2_cubic[3, 0]], "Root_C2_P3_Z": [root_c2_cubic[3, 1]],

        # Tip Curve 1
        "Tip_C1_P0_X": [tip_c1_top_3d[0, 0]], "Tip_C1_P0_Y": [tip_c1_top_3d[0, 1]], "Tip_C1_P0_Z": [tip_c1_top_3d[0, 2]],
        "Tip_C1_P1_X": [tip_c1_top_3d[1, 0]], "Tip_C1_P1_Y": [tip_c1_top_3d[1, 1]], "Tip_C1_P1_Z": [tip_c1_top_3d[1, 2]],
        "Tip_C1_P2_X": [tip_c1_top_3d[2, 0]], "Tip_C1_P2_Y": [tip_c1_top_3d[2, 1]], "Tip_C1_P2_Z": [tip_c1_top_3d[2, 2]],
        "Tip_C1_P3_X": [tip_c1_top_3d[3, 0]], "Tip_C1_P3_Y": [tip_c1_top_3d[3, 1]], "Tip_C1_P3_Z": [tip_c1_top_3d[3, 2]],

        # Tip Curve 2
        "Tip_C2_P0_X": [tip_c2_top_3d[0, 0]], "Tip_C2_P0_Y": [tip_c2_top_3d[0, 1]], "Tip_C2_P0_Z": [tip_c2_top_3d[0, 2]],
        "Tip_C2_P1_X": [tip_c2_top_3d[1, 0]], "Tip_C2_P1_Y": [tip_c2_top_3d[1, 1]], "Tip_C2_P1_Z": [tip_c2_top_3d[1, 2]],
        "Tip_C2_P2_X": [tip_c2_top_3d[2, 0]], "Tip_C2_P2_Y": [tip_c2_top_3d[2, 1]], "Tip_C2_P2_Z": [tip_c2_top_3d[2, 2]],
        "Tip_C2_P3_X": [tip_c2_top_3d[3, 0]], "Tip_C2_P3_Y": [tip_c2_top_3d[3, 1]], "Tip_C2_P3_Z": [tip_c2_top_3d[3, 2]],

        # Hub (X/Z naming)
        "Hub_P0_X": [hub_cp_az[0, 0]], "Hub_P0_Z": [hub_cp_az[0, 1]],
        "Hub_P1_X": [hub_cp_az[1, 0]], "Hub_P1_Z": [hub_cp_az[1, 1]],
        "Hub_P2_X": [hub_cp_az[2, 0]], "Hub_P2_Z": [hub_cp_az[2, 1]],
        "Hub_P3_X": [hub_cp_az[3, 0]], "Hub_P3_Z": [hub_cp_az[3, 1]],

        # Continuity residuals
        "Join_C0_residual": [c0_res],
        "Join_C1_residual": [c1_res],
        "Join_C2_residual": [c2_res],

        # Abaqus (current selection)
        "Abaqus_Target": [abaqus_target],
        "Abaqus_u": [abaqus_u],
        "Abaqus_LoadMag": [load_mag],
        "LoadPoint_X": [load_point_3d[0]],
        "LoadPoint_Y": [load_point_3d[1]],
        "LoadPoint_Z": [load_point_3d[2]],
        "LoadDir_X": [normal_unit_3d[0]],
        "LoadDir_Y": [normal_unit_3d[1]],
        "LoadDir_Z": [normal_unit_3d[2]],

        # CFD
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
        variant_table.to_excel(writer, sheet_name="BladeVariants", index=False)

    design_table_bytes = buf_dt.getvalue()

    # Config template file: if present on disk, use it. Else generate a basic one.
    template_bytes = read_template_bytes()
    if template_bytes is None:
        buf_cfg = BytesIO()
        with pd.ExcelWriter(buf_cfg, engine="xlsxwriter") as writer:
            default_variant_table().to_excel(writer, sheet_name="Table1_BladeVariants", index=False)
            pd.DataFrame({"Name": ["NumBlades", "ScaleX", "ScaleZ", "TwistTotalDeg", "TwistSign", "AbaqusLoadMag", "AbaqusU", "AbaqusTarget"],
                          "Value": [3, 0.8, 0.8, 15.0, +1, 1000.0, 0.5, "Tip_C2_Top"]}).to_excel(writer, sheet_name="Parameters", index=False)
            # Optional sheets used by app if you fill them:
            pd.DataFrame({"Section": ["Root"]*4, "Curve": [1]*4, "PointIndex": [0,1,2,3],
                          "X": [1.0, 1.0, 0.8, 0.6], "Z": [0.0, 0.06, 0.08, 0.075]}).to_excel(writer, sheet_name="BezierControlPoints", index=False)
            pd.DataFrame({"PointIndex": [0,1,2,3], "X": [-0.4, 0.3, 13.0, 2.0], "Z": [1.0, 1.0, 0.8, 0.0]}).to_excel(writer, sheet_name="HubControlPoints", index=False)
        template_bytes = buf_cfg.getvalue()

    # Abaqus Python script stub
    script_lines = f"""# Abaqus script stub generated by CAE4 Streamlit tool
# Target: {abaqus_target}, u={abaqus_u:.4f}
# Load point: {tuple(load_point_3d)}
# Load dir (unit): {tuple(normal_unit_3d)}
# Load magnitude: {load_mag:.3f} N

from abaqus import *
from abaqusConstants import *
import regionToolset

MODEL_NAME = 'BladeModel'
PART_NAME  = 'BladePart'
INSTANCE_NAME = 'BladeInstance'
STEP_GEOMETRY_FILE = 'blade_geometry.step'  # TODO: replace with your geometry filename

load_point = ({load_point_3d[0]:.6f}, {load_point_3d[1]:.6f}, {load_point_3d[2]:.6f})
load_dir   = ({normal_unit_3d[0]:.6f}, {normal_unit_3d[1]:.6f}, {normal_unit_3d[2]:.6f})
load_mag   = {load_mag:.6f}  # N

# NOTE:
# This stub assumes you already imported geometry into the model and the part/instance names match.
# You may need to adjust PART_NAME / INSTANCE_NAME depending on your import workflow.

# Example: open an existing model OR create/import based on your workflow.
# mdb.ModelFromInputFile(name=MODEL_NAME, inputFileName=STEP_GEOMETRY_FILE)

model = mdb.models[MODEL_NAME]
part = model.parts[PART_NAME]

a = model.rootAssembly
try:
    inst = a.instances[INSTANCE_NAME]
except KeyError:
    inst = a.Instance(name=INSTANCE_NAME, part=part, dependent=ON)

rp = a.ReferencePoint(point=load_point)
rp_region = regionToolset.Region(referencePoints=(rp,))

if 'LoadStep' not in model.steps.keys():
    model.StaticStep(name='LoadStep', previous='Initial')

model.ConcentratedLoad(
    name='TipLoad',
    createStepName='LoadStep',
    region=rp_region,
    cf1=load_mag * load_dir[0],
    cf2=load_mag * load_dir[1],
    cf3=load_mag * load_dir[2],
)

print('Script setup complete. Verify part/instance names, BCs, and meshing before running.')
"""
    script_bytes = script_lines.encode("utf-8")

    # Place download buttons in the sidebar downloads expander (as requested)
    with st.sidebar:
        with downloads_placeholder.container():
            with st.expander("Downloads", expanded=False):
                st.download_button(
                    "📥 Download CATIA design table (Excel)",
                    data=design_table_bytes,
                    file_name="blade_hub_cfd_design_table.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.download_button(
                    "📥 Download config template (Excel)",
                    data=template_bytes,
                    file_name="config_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.download_button(
                    "📥 Download Abaqus Python script stub (.py)",
                    data=script_bytes,
                    file_name="abaqus_blade_load_stub.py",
                    mime="text/x-python",
                )
                # Optional exports re-added
                hub_csv = hub_df_out_for_design.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download hub control points (CSV)",
                    data=hub_csv,
                    file_name="hub_control_points.csv",
                    mime="text/csv",
                )
                cfd_box_df = pd.DataFrame(
                    {"Param": ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax", "Chord"],
                     "Value_m": [cfd_x_min, cfd_x_max, cfd_y_min, cfd_y_max, cfd_z_min, cfd_z_max, chord_length]}
                )
                cfd_csv = cfd_box_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CFD domain parameters (CSV)",
                    data=cfd_csv,
                    file_name="cfd_domain_parameters.csv",
                    mime="text/csv",
                )
                sampled_csv = sampled_blade_points_csv(
                    root_c1_cubic, root_c2_cubic,
                    tip_c1_top_2d, tip_c2_top_2d,
                    tip_c1_bottom_2d, tip_c2_bottom_2d,
                    blade_length_m
                )
                st.download_button(
                    "Download sampled blade points (CSV)",
                    data=sampled_csv,
                    file_name="blade_sampled_points.csv",
                    mime="text/csv",
                )

    # ---------- Blade geometry tab ----------
    with tab_blade:
        st.header("Blade geometry")

        st.markdown(
            f"**Effective parameters used:**  "
            f"`Num blades` = {num_blades}, "
            f"`Blade length` = {blade_length_m:.2f} m, "
            f"`Thickness` = {blade_thickness_m*1000:.2f} mm, "
            f"`ScaleX` = {scale_x:.3f}, "
            f"`ScaleZ` = {scale_z:.3f}, "
            f"`Twist` = {twist_total_deg:.2f}° total ({twist_deg_per_m:.4f} °/m), sign={twist_sign:+.0f}."
        )

        st.subheader("Root cross-section – top surface")
        st.plotly_chart(make_root_top_plot(root_c1_cubic, root_c2_cubic), use_container_width=True)

        st.subheader("Root & tip cross-sections – top and mirrored bottom")
        fig = make_root_and_tip_plot(
            root_c1_cubic, root_c2_cubic,
            tip_c1_top_2d, tip_c2_top_2d,
            tip_c1_bottom_2d, tip_c2_bottom_2d,
            show_root=bool(st.session_state.show_root),
            show_tip=bool(st.session_state.show_tip),
            x_range=x_range, z_range=z_range,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Continuity check at Curve 1 → Curve 2 join (Root top)")
        st.table(pd.DataFrame(
            {"Residual": ["C0", "C1", "C2"], "Value": [c0_res, c1_res, c2_res]}
        ))

        st.subheader("Tip Curve (selected Abaqus target) – point and normal")
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

        st.subheader("Control points for manufacturing – Blade")
        st.markdown("**Root – cubic control points (top half, Y = 0)**")
        st.dataframe(pd.concat([root_c1_df_out, root_c2_df_out], ignore_index=True))

        st.markdown("**Tip – cubic control points (top half, Y = Blade length)**")
        st.dataframe(pd.concat([tip_c1_df_out, tip_c2_df_out], ignore_index=True))

    # ---------- 3D preview tab ----------
    with tab_3d:
        st.header("3D preview – Blade")
        st.markdown(
            "Simple 3D visualisation of the blade root and tip sections, with a few spanwise ribs "
            "to give an impression of twist and taper."
        )
        st.plotly_chart(
            make_blade_3d_plot(root_c1_cubic, root_c2_cubic, tip_c1_top_2d, tip_c2_top_2d, blade_length_m),
            use_container_width=True
        )

    # ---------- Hub geometry tab ----------
    with tab_hub:
        st.header("Hub geometry")
        st.markdown(f"Hub profile control points source: **{hub_source_label}**")
        st.plotly_chart(make_hub_plot(hub_cp_az, hub_plane_label=hub_plane_label), use_container_width=True)

        st.markdown("**Hub control points**")
        st.dataframe(hub_df_out_for_design)

    # ---------- CFD domain tab ----------
    with tab_cfd:
        st.header("CFD domain sizing")
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
                    chord_length, cfd_x_min, cfd_x_max, cfd_y_min, cfd_y_max, cfd_z_min, cfd_z_max
                ],
            }
        )
        st.table(cfd_df)

    # ---------- Abaqus config tab ----------
    with tab_abaqus:
        st.header("Abaqus configuration helpers")

        st.subheader("Load point and normal vector (from selected target in sidebar)")
        st.table(df_load)

        st.subheader("Abaqus Python script stub (preview)")
        st.code(script_lines, language="python")


if __name__ == "__main__":
    main()
