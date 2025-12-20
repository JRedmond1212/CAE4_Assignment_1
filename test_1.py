import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
import os
import hashlib
from typing import Tuple, List, Optional

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
    candidate_sheets = candidate_sheets + [s for s in xls.sheet_names if s not in candidate_sheets]

    def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in ["numblades", "num_blades", "blades", "nblades", "n_blades"]:
                col_map[c] = "NumBlades"
            if cl in ["blade_length", "bladelength", "length", "blade_length_m", "bladelength_m", "length_m"]:
                col_map[c] = "BladeLength_m"
            if cl in ["thickness", "blade_thickness", "bladethickness", "thickness_m", "blade_thickness_m"]:
                col_map[c] = "Thickness_m"
        return df.rename(columns=col_map).copy()

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


def _add_tip_c2_u_labels(fig, tip_c2_top_2d):
    u_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    xs, zs, texts = [], [], []
    for u in u_vals:
        pt, _ = bezier_point_and_tangent(tip_c2_top_2d, float(u))
        xs.append(float(pt[0]))
        zs.append(float(pt[1]))
        texts.append(f"u={u:.1f}")

    fig.add_trace(go.Scatter(
        x=xs, y=zs,
        mode="markers+text",
        name="Tip C2 u-ratio",
        text=texts,
        textposition="top center",
        marker=dict(size=6, color=COLOR_TIP_C2),
        hovertemplate="Tip C2 label<br>%{text}<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        showlegend=True,
    ))


def _add_force_arrow_2d(fig, load_xz, dir_xz_unit, arrow_len):
    x0, z0 = float(load_xz[0]), float(load_xz[1])
    dx, dz = float(dir_xz_unit[0]), float(dir_xz_unit[1])

    x1 = x0 + arrow_len * dx
    z1 = z0 + arrow_len * dz

    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[z0, z1],
        mode="lines+markers",
        name="Applied force (proj. X–Z)",
        line=dict(width=4),
        marker=dict(size=6),
        hovertemplate="Force arrow (XZ)<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=[x0], y=[z0],
        mode="markers",
        name="Load point (XZ)",
        marker=dict(size=10, symbol="x"),
        hovertemplate="Load point (XZ)<br>x=%{x:.4f}<br>z=%{y:.4f}<extra></extra>",
        showlegend=True,
    ))


def make_root_and_tip_plot(
    root_c1, root_c2,
    tip_c1_top_2d, tip_c2_top_2d,
    tip_c1_bottom_2d, tip_c2_bottom_2d,
    show_root, show_tip,
    x_range, z_range,
    show_tip_c2_u_labels: bool = True,
    force_overlay_enabled: bool = False,
    force_load_xz: Optional[Tuple[float, float]] = None,
    force_dir_xz_unit: Optional[Tuple[float, float]] = None,
    force_arrow_len: float = 0.0,
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

        if show_tip_c2_u_labels and tip_c2_top_2d is not None:
            _add_tip_c2_u_labels(fig, tip_c2_top_2d)

    if force_overlay_enabled and force_load_xz is not None and force_dir_xz_unit is not None and force_arrow_len > 0:
        _add_force_arrow_2d(fig, force_load_xz, force_dir_xz_unit, force_arrow_len)

    return apply_common_layout_tweaks(
        fig,
        "Blade Outer Profile (Root & Tip, mirrored about X-axis)",
        x_range=x_range,
        z_range=z_range,
        x_label="x (chord-wise)",
        z_label="z (thickness direction)",
    )


def make_hub_plot(hub_cp_az, hub_plane_label="X–Z"):
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
    fig.add_trace(go.Scatter3d(x=x1r, y=np.zeros_like(x1r), z=z1r_top, mode="lines",
                               name="Root C1 top", line=dict(color=COLOR_ROOT_C1)))
    fig.add_trace(go.Scatter3d(x=x2r, y=np.zeros_like(x2r), z=z2r_top, mode="lines",
                               name="Root C2 top", line=dict(color=COLOR_ROOT_C2)))
    fig.add_trace(go.Scatter3d(x=x1r, y=np.zeros_like(x1r), z=-z1r_top, mode="lines",
                               name="Root C1 bottom", line=dict(color=COLOR_ROOT_C1)))
    fig.add_trace(go.Scatter3d(x=x2r, y=np.zeros_like(x2r), z=-z2r_top, mode="lines",
                               name="Root C2 bottom", line=dict(color=COLOR_ROOT_C2)))

    # Tip (y=L)
    _, x1t, z1t_top = general_bezier_curve(tip_c1_top_2d)
    _, x2t, z2t_top = general_bezier_curve(tip_c2_top_2d)
    fig.add_trace(go.Scatter3d(x=x1t, y=np.ones_like(x1t) * blade_length_m, z=z1t_top, mode="lines",
                               name="Tip C1 top", line=dict(color=COLOR_TIP_C1)))
    fig.add_trace(go.Scatter3d(x=x2t, y=np.ones_like(x2t) * blade_length_m, z=z2t_top, mode="lines",
                               name="Tip C2 top", line=dict(color=COLOR_TIP_C2)))
    fig.add_trace(go.Scatter3d(x=x1t, y=np.ones_like(x1t) * blade_length_m, z=-z1t_top, mode="lines",
                               name="Tip C1 bottom", line=dict(color=COLOR_TIP_C1)))
    fig.add_trace(go.Scatter3d(x=x2t, y=np.ones_like(x2t) * blade_length_m, z=-z2t_top, mode="lines",
                               name="Tip C2 bottom", line=dict(color=COLOR_TIP_C2)))

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


def add_force_to_3d_fig(fig, load_point_3d, force_dir_unit_3d, chord_length, label="Applied force"):
    p = np.asarray(load_point_3d, dtype=float)
    d = np.asarray(force_dir_unit_3d, dtype=float)
    dmag = float(np.linalg.norm(d))
    if dmag <= 0:
        return fig

    d = d / dmag
    arrow_len = 0.25 * float(chord_length) if chord_length > 0 else 0.25
    p2 = p + arrow_len * d

    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode="markers",
        name="Load point (target)",
        marker=dict(size=6, symbol="x"),
        hovertemplate="Load point<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[p[0], p2[0]],
        y=[p[1], p2[1]],
        z=[p[2], p2[2]],
        mode="lines+markers",
        name=label,
        line=dict(width=6),
        marker=dict(size=3),
        hovertemplate="Force arrow<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
    ))
    return fig


# =========================================================
# Export helpers (sampled points)
# =========================================================

def sampled_blade_points_csv(root_c1, root_c2, tip_c1_top_2d, tip_c2_top_2d, tip_c1_bot_2d, tip_c2_bot_2d, blade_length_m):
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

    add_curve("Root", "Top", "C1", 0.0, root_c1)
    add_curve("Root", "Top", "C2", 0.0, root_c2)
    add_curve("Root", "Bottom", "C1", 0.0, root_c1 * np.array([1.0, -1.0]))
    add_curve("Root", "Bottom", "C2", 0.0, root_c2 * np.array([1.0, -1.0]))

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

    # We keep the UI target list, but the NEW Abaqus script is designed
    # specifically to load "Tip_C2_Top" by partitioning its tip edge.
    st.session_state.abaqus_target = "Tip_C2_Top"

    # Abaqus model database defaults
    st.session_state.abaqus_model_name = "Model-1"
    st.session_state.abaqus_part_name = "Model-1"      # your imported part name
    st.session_state.abaqus_instance_name = "Blade-1"
    st.session_state.abaqus_material_name = "BladeMaterial"
    st.session_state.abaqus_E = 70e9
    st.session_state.abaqus_nu = 0.33
    st.session_state.abaqus_mesh_seed = 0.50          # "meters-ish"; Abaqus script tolerances are bbox-driven
    st.session_state.abaqus_job_name = "BladeJob"
    st.session_state.abaqus_step_name = "Step-1"

    st.session_state.last_config_hash = None


def apply_config_once_if_new(uploaded_bytes: bytes) -> Tuple[bool, List[str], pd.DataFrame]:
    warnings = []
    applied = False
    variant_table = default_variant_table()

    if uploaded_bytes is None:
        return applied, warnings, variant_table

    cfg_hash = file_hash_bytes(uploaded_bytes)
    if st.session_state.last_config_hash == cfg_hash:
        return applied, warnings, variant_table

    st.session_state.last_config_hash = cfg_hash
    applied = True

    try:
        xls = pd.ExcelFile(BytesIO(uploaded_bytes))
    except Exception:
        warnings.append("Config upload could not be read as an Excel file. Using current sidebar values.")
        return applied, warnings, variant_table

    vt = parse_variant_table_from_excel(xls)
    if vt is not None:
        variant_table = vt
        new_opts = list(map(int, variant_table["NumBlades"].tolist()))
        if len(new_opts) >= 1:
            st.session_state.blade_options = new_opts
            if int(st.session_state.num_blades) not in new_opts:
                st.session_state.num_blades = int(new_opts[0])
    else:
        warnings.append("No blade variant table found in config; using default blade options (3,4,5).")

    params = load_parameters(BytesIO(uploaded_bytes))
    if params:
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

        # Abaqus params if present
        if "AbaqusLoadMag" in params:
            try_set_float("abaqus_load_mag", "AbaqusLoadMag")
        if "AbaqusU" in params:
            try_set_float("abaqus_u", "AbaqusU")
        if "AbaqusTarget" in params:
            st.session_state.abaqus_target = str(params["AbaqusTarget"])

        # Abaqus material overrides
        if "AbaqusMaterialName" in params:
            st.session_state.abaqus_material_name = str(params["AbaqusMaterialName"])
        if "AbaqusE" in params:
            try_set_float("abaqus_E", "AbaqusE")
        if "AbaqusNu" in params:
            try_set_float("abaqus_nu", "AbaqusNu")
        if "AbaqusMeshSeed" in params:
            try_set_float("abaqus_mesh_seed", "AbaqusMeshSeed")

        # Model naming overrides
        if "AbaqusModelName" in params:
            st.session_state.abaqus_model_name = str(params["AbaqusModelName"])
        if "AbaqusPartName" in params:
            st.session_state.abaqus_part_name = str(params["AbaqusPartName"])
        if "AbaqusInstanceName" in params:
            st.session_state.abaqus_instance_name = str(params["AbaqusInstanceName"])
        if "AbaqusStepName" in params:
            st.session_state.abaqus_step_name = str(params["AbaqusStepName"])
        if "AbaqusJobName" in params:
            st.session_state.abaqus_job_name = str(params["AbaqusJobName"])

    cp_df = load_bezier_control_points(BytesIO(uploaded_bytes))
    if cp_df is not None:
        sub = cp_df[(cp_df.get("Section") == "Root") & (cp_df.get("Curve") == 1)]
        if not sub.empty and {"X", "Z"}.issubset(sub.columns):
            sub_sorted = sort_curve_points(sub)
            X = sub_sorted["X"].to_numpy()
            Z = sub_sorted["Z"].to_numpy()
            if not (np.isnan(X).any() or np.isnan(Z).any()) and X.shape[0] >= 4:
                st.session_state.c1_p0_x = float(X[0]); st.session_state.c1_p0_z = float(Z[0])
                st.session_state.c1_p1_x = float(X[1]); st.session_state.c1_p1_z = float(Z[1])
                st.session_state.c1_p2_x = float(X[2]); st.session_state.c1_p2_z = float(Z[2])
                st.session_state.c1_p3_x = float(X[3]); st.session_state.c1_p3_z = float(Z[3])
            else:
                warnings.append("BezierControlPoints Root/Curve1 found but invalid; ignored.")

    hub_cp, _hub_plane = load_hub_control_points(BytesIO(uploaded_bytes))
    if hub_cp is not None and hub_cp.shape[0] >= 4:
        st.session_state.hub_p0_x = float(hub_cp[0, 0]); st.session_state.hub_p0_z = float(hub_cp[0, 1])
        st.session_state.hub_p1_x = float(hub_cp[1, 0]); st.session_state.hub_p1_z = float(hub_cp[1, 1])
        st.session_state.hub_p2_x = float(hub_cp[2, 0]); st.session_state.hub_p2_z = float(hub_cp[2, 1])
        st.session_state.hub_p3_x = float(hub_cp[3, 0]); st.session_state.hub_p3_z = float(hub_cp[3, 1])
    elif hub_cp is None:
        warnings.append("No valid HubControlPoints found in config; using current hub defaults.")

    return applied, warnings, variant_table


# =========================================================
# Abaqus script generator (POST-IMPORT) - TIP CURVE2 TOP PARTITION-BY-U
# =========================================================

def generate_abaqus_post_import_script_tip_curve2_partition_u(
    model_name: str,
    part_name: str,
    instance_name: str,
    material_name: str,
    E: float,
    nu: float,
    shell_thickness: float,
    mesh_seed: float,
    step_name: str,
    job_name: str,
    # from Streamlit geometry:
    load_u: float,
    load_mag: float,
    # hint point near tip curve2-top at chosen u, in (x,z)
    load_point_hint_xz: Tuple[float, float],
    # force direction unit (x,y,z) computed from the Bezier tangent/normal (y is 0 here)
    force_dir_unit_xyz: Tuple[float, float, float],
    flip_dir: bool,
):
    hx, hz = float(load_point_hint_xz[0]), float(load_point_hint_xz[1])
    dx, dy, dz = float(force_dir_unit_xyz[0]), float(force_dir_unit_xyz[1]), float(force_dir_unit_xyz[2])

    # Apply flip in script so the generated code is self-consistent
    if flip_dir:
        dx, dy, dz = -dx, -dy, -dz

    # Force vector in global coords
    fx, fy, fz = load_mag * dx, load_mag * dy, load_mag * dz

    # NOTE:
    # - We DO NOT use Streamlit's "blade_length_m" directly inside Abaqus selection.
    # - We detect tip_y from the PART bbox (robust to mm vs m).
    # - We find "tip plane edges" by bounding box y ~ tip_y.
    # - We choose the best edge by closeness to the hint point (x,z).
    # - We partition that edge by param u (PartitionEdgeByParam).
    # - Then we find the created vertex (closest to hint) and apply load to closest mesh node to that vertex.
    return f"""# -*- coding: utf-8 -*-
# Abaqus/CAE script (POST-IMPORT) generated by Streamlit
# Goal: Apply a concentrated load on TIP Curve2 TOP at parameter u by:
#   1) detect tip plane Y from bbox
#   2) identify the correct tip edge using a (x,z) hint
#   3) partition the edge by param u (PartitionEdgeByParam)
#   4) mesh
#   5) apply load at the node closest to the partition vertex
#
# IMPORTANT: This script intentionally has NO "fallback continue" logic.
# If a stage fails, it raises and stops so you can fix the root cause.

from abaqus import mdb
from abaqusConstants import *
import regionToolset
import mesh

MODEL_NAME    = r\"{model_name}\"
PART_NAME     = r\"{part_name}\"
INSTANCE_NAME = r\"{instance_name}\"
MATERIAL_NAME = r\"{material_name}\"
STEP_NAME     = r\"{step_name}\"
JOB_NAME      = r\"{job_name}\"

E       = {E:.16g}
NU      = {nu:.16g}
SHELL_T = {shell_thickness:.16g}
MESH_SEED = {mesh_seed:.16g}

LOAD_U = {float(load_u):.16g}
LOAD_POINT_HINT_XZ = ({hx:.16g}, {hz:.16g})

FORCE_VEC = ({fx:.16g}, {fy:.16g}, {fz:.16g})

def log(msg):
    print('>>> ' + str(msg))

def fail(msg):
    raise RuntimeError(str(msg))

def get_model():
    if MODEL_NAME not in mdb.models:
        fail(\"Model '%s' not found in mdb.models\" % MODEL_NAME)
    return mdb.models[MODEL_NAME]

def get_part(model):
    if PART_NAME not in model.parts:
        fail(\"Part '%s' not found in model.parts\" % PART_NAME)
    return model.parts[PART_NAME]

def get_or_create_instance(model, part_obj):
    a = model.rootAssembly
    if INSTANCE_NAME in a.instances:
        return a.instances[INSTANCE_NAME]

    if len(a.instances) > 0:
        # Reuse first instance if user already has one, to avoid duplicates
        inst = a.instances.values()[0]
        log(\"Instance '%s' not found; reusing first instance '%s'.\" % (INSTANCE_NAME, inst.name))
        return inst

    log(\"No instances found. Creating instance %s.\" % INSTANCE_NAME)
    a.DatumCsysByDefault(CARTESIAN)
    inst = a.Instance(name=INSTANCE_NAME, part=part_obj, dependent=ON)
    return inst

def ensure_material_and_section(model, part_obj):
    log(\"Ensuring material and shell section...\")
    if MATERIAL_NAME not in model.materials:
        mat = model.Material(name=MATERIAL_NAME)
        mat.Elastic(table=((E, NU),))
        log(\"Created material: %s\" % MATERIAL_NAME)

    sec_name = \"BladeShellSection\"
    if sec_name not in model.sections:
        model.HomogeneousShellSection(
            name=sec_name,
            material=MATERIAL_NAME,
            thicknessType=UNIFORM,
            thickness=SHELL_T,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            useDensity=OFF,
            integrationRule=SIMPSON,
            numIntPts=5,
        )
        log(\"Created section: %s\" % sec_name)

    faces = part_obj.faces
    if len(faces) == 0:
        fail(\"Part has no faces. Is this a shell/surface part?\")
    region = regionToolset.Region(faces=faces)
    part_obj.SectionAssignment(
        region=region,
        sectionName=sec_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField=\"\",
        thicknessAssignment=FROM_SECTION
    )
    log(\"Assigned section to all faces.\")

def ensure_step(model):
    log(\"Ensuring analysis step...\")
    if STEP_NAME in model.steps:
        log(\"Step exists: %s\" % STEP_NAME)
        return
    model.StaticStep(name=STEP_NAME, previous=\"Initial\")
    log(\"Created step: %s\" % STEP_NAME)

def detect_tip_y_and_tol(part_obj):
    log(\"Detecting tip Y (span direction) from geometry...\")
    bb = part_obj.getBoundingBox()
    lo = bb['low']; hi = bb['high']
    x0,y0,z0 = lo[0],lo[1],lo[2]
    x1,y1,z1 = hi[0],hi[1],hi[2]
    log(\"BBox: x=[%s, %s], y=[%s, %s], z=[%s, %s]\" % (x0,x1,y0,y1,z0,z1))
    tip_y = y1
    tol_y = max(1e-6, 1e-4*abs(tip_y) if abs(tip_y) > 0 else 1e-6)
    log(\"Detected tip_y = y_max = %s, tol_y = %s\" % (tip_y, tol_y))
    if abs(tip_y) > 1000.0:
        log(\"NOTE: tip_y is large -> model likely in mm (or similar). That's OK; script uses bbox-based tolerances.\")
    return tip_y, tol_y, (x0,x1,y0,y1,z0,z1)

def apply_root_pinned_bc(model, inst, tol_y):
    log(\"Stage: apply root pinned BC\")
    edges = inst.edges.getByBoundingBox(
        xMin=-1e99, xMax=1e99,
        yMin=-tol_y, yMax=+tol_y,
        zMin=-1e99, zMax=1e99
    )
    if len(edges) == 0:
        fail(\"No root edges found near y=0 (within tol_y). Check coordinate system.\")
    set_name = \"SET_ROOT_EDGES\"
    if set_name in inst.sets:
        del inst.sets[set_name]
    inst.Set(name=set_name, edges=edges)
    region = inst.sets[set_name]
    bc_name = \"BC_ROOT_PINNED\"
    if bc_name in model.boundaryConditions:
        del model.boundaryConditions[bc_name]
    model.DisplacementBC(
        name=bc_name,
        createStepName=\"Initial\",
        region=region,
        u1=0.0, u2=0.0, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET,
        distributionType=UNIFORM,
        fieldName=\"\",
        localCsys=None
    )
    log(\"Applied pinned BC on %d root edges.\" % len(edges))

def _edge_midpoint(edge):
    # Edge.pointOn is a tuple of tuples: e.g. ((x,y,z),)
    p = edge.pointOn[0]
    return (float(p[0]), float(p[1]), float(p[2]))

def find_tip_curve2_top_edge(inst, tip_y, tol_y):
    log(\"Stage: identify Curve2-top tip edge\")
    log(\"Finding candidate edges on the tip plane (y ~ tip_y)...\")
    tip_edges = inst.edges.getByBoundingBox(
        xMin=-1e99, xMax=1e99,
        yMin=tip_y - tol_y, yMax=tip_y + tol_y,
        zMin=-1e99, zMax=1e99
    )
    if len(tip_edges) == 0:
        fail(\"No edges found on tip plane. Check that blade span is along +Y.\")

    log(\"Tip edges found: %d\" % len(tip_edges))

    # Choose edge closest (in XZ) to hint point.
    hx, hz = LOAD_POINT_HINT_XZ
    best_e = None
    best_d2 = None

    for e in tip_edges:
        xm, ym, zm = _edge_midpoint(e)
        d2 = (xm - hx)*(xm - hx) + (zm - hz)*(zm - hz)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_e = e

    if best_e is None:
        fail(\"Failed to select a tip edge (unexpected).\")

    log(\"Selected tip edge by XZ-hint. best_d2=%s\" % best_d2)
    return best_e

def partition_edge_by_u(part_obj, inst, edge_inst, u):
    log(\"Stage: partition edge at LOAD_U\")
    if u <= 0.0 or u >= 1.0:
        fail(\"LOAD_U must be strictly between 0 and 1 for partitioning. Got %s\" % u)

    # We must partition the PART edge, not the INSTANCE edge.
    # So, locate the corresponding part edge using a point on the instance edge.
    p = edge_inst.pointOn[0]
    x,y,z = float(p[0]), float(p[1]), float(p[2])

    # findAt expects a sequence of coordinates
    try:
        edge_part = part_obj.edges.findAt(((x,y,z),))
    except Exception as ex:
        fail(\"Could not map instance edge to part edge using findAt. Error: %s\" % ex)

    log(\"Partitioning part edge by param u=%s ...\" % u)
    try:
        part_obj.PartitionEdgeByParam(edges=edge_part, parameter=u)
    except Exception as ex:
        fail(\"PartitionEdgeByParam failed. Error: %s\" % ex)

    log(\"PartitionEdgeByParam succeeded.\")

def mesh_part(part_obj):
    log(\"Stage: mesh part\")
    try:
        part_obj.setMeshControls(regions=part_obj.faces, elemShape=QUAD, technique=FREE)
    except Exception:
        pass

    if MESH_SEED <= 0.0:
        fail(\"MESH_SEED must be > 0\")

    part_obj.seedPart(size=MESH_SEED, deviationFactor=0.1, minSizeFactor=0.1)

    elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=S3,  elemLibrary=STANDARD)
    part_obj.setElementType(regions=(part_obj.faces,), elemTypes=(elemType1, elemType2))

    part_obj.generateMesh()
    log(\"Meshing complete.\")

def apply_load_at_partition_vertex(model, inst, tip_y, tol_y):
    log(\"Stage: apply load at partition vertex (closest to hint)\")
    # After partition + mesh, the partition creates a vertex on that tip edge.
    # We find the vertex on the tip plane closest to hint, then apply load at closest node to that vertex.

    verts = inst.vertices.getByBoundingBox(
        xMin=-1e99, xMax=1e99,
        yMin=tip_y - tol_y, yMax=tip_y + tol_y,
        zMin=-1e99, zMax=1e99
    )
    if len(verts) == 0:
        fail(\"No vertices found on tip plane after partition. Partition probably did not create a vertex where expected.\")

    hx, hz = LOAD_POINT_HINT_XZ
    best_v = None
    best_d2 = None
    for v in verts:
        p = v.pointOn[0]
        x,z = float(p[0]), float(p[2])
        d2 = (x-hx)*(x-hx) + (z-hz)*(z-hz)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_v = v

    if best_v is None:
        fail(\"Failed to choose a tip vertex (unexpected).\")

    pv = best_v.pointOn[0]
    vx, vy, vz = float(pv[0]), float(pv[1]), float(pv[2])
    log(\"Chosen vertex: (x,y,z)=(%s,%s,%s)  d2=%s\" % (vx,vy,vz,best_d2))

    # Must have mesh nodes
    nodes = inst.nodes
    if len(nodes) == 0:
        fail(\"No nodes on instance. Meshing likely failed.\")
    res = nodes.getClosest(coordinates=(vx, vy, vz))
    if res is None or len(res) == 0:
        fail(\"getClosest returned no nodes near partition vertex.\")

    first = res[0]
    try:
        node_obj = first[0]
    except Exception:
        node_obj = first

    set_name = \"SET_LOAD_NODE\"
    if set_name in inst.sets:
        del inst.sets[set_name]
    inst.Set(name=set_name, nodes=nodes.sequenceFromLabels((node_obj.label,)))
    region = inst.sets[set_name]

    load_name = \"LOAD_F\"
    if load_name in model.loads:
        del model.loads[load_name]

    model.ConcentratedForce(
        name=load_name,
        createStepName=STEP_NAME,
        region=region,
        cf1=FORCE_VEC[0],
        cf2=FORCE_VEC[1],
        cf3=FORCE_VEC[2],
        distributionType=UNIFORM,
        field=\"\",
        localCsys=None
    )
    log(\"Applied concentrated force at node label %d\" % node_obj.label)

def make_job_and_run(model):
    log(\"Stage: job submit\")
    if JOB_NAME in mdb.jobs:
        del mdb.jobs[JOB_NAME]
    mdb.Job(
        name=JOB_NAME,
        model=MODEL_NAME,
        type=ANALYSIS,
        numCpus=1,
        numDomains=1,
        memory=90,
        memoryUnits=PERCENTAGE,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF
    )
    mdb.jobs[JOB_NAME].submit(consistencyChecking=OFF)
    mdb.jobs[JOB_NAME].waitForCompletion()
    log(\"Job completed.\")

def main():
    log(\"===== START POST-IMPORT BLADE SCRIPT =====\")
    model = get_model()
    part_obj = get_part(model)
    inst = get_or_create_instance(model, part_obj)
    log(\"Using model='%s', part='%s', instance='%s'\" % (MODEL_NAME, part_obj.name, inst.name))

    ensure_material_and_section(model, part_obj)
    ensure_step(model)

    # Detect tip Y and tolerances from bbox (robust to mm vs m)
    tip_y, tol_y, _bbox = detect_tip_y_and_tol(part_obj)

    # Identify the correct TIP Curve2-top edge from hint point (must succeed)
    edge_tip = find_tip_curve2_top_edge(inst, tip_y, tol_y)

    # Partition that edge by parameter u (must succeed)
    partition_edge_by_u(part_obj, inst, edge_tip, LOAD_U)

    # Mesh after partition
    mesh_part(part_obj)

    # Regenerate assembly after meshing
    model.rootAssembly.regenerate()

    # Apply root BC
    apply_root_pinned_bc(model, inst, tol_y)

    # Apply load at partition vertex (closest to hint)
    apply_load_at_partition_vertex(model, inst, tip_y, tol_y)

    # Run job
    make_job_and_run(model)

    log(\"===== END POST-IMPORT BLADE SCRIPT =====\")

main()
"""


# =========================================================
# Streamlit app
# =========================================================

def main():
    st.set_page_config(page_title="CAE4 Blade & Hub Tool", layout="wide")
    ensure_state_defaults()

    # ---------- Sidebar: upload early ----------
    with st.sidebar:
        with st.expander("Config", expanded=False):
            st.markdown("**Excel configuration (optional)**")
            uploaded_file = st.file_uploader(
                "Upload Excel configuration (optional)",
                type=["xlsx"],
                key="excel_upload",
            )
        uploaded_bytes = uploaded_file.read() if uploaded_file is not None else None

    config_applied, config_warnings, variant_table = apply_config_once_if_new(uploaded_bytes)

    # ---------- Sidebar: widgets ----------
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

            blade_opts = st.session_state.get("blade_options", [3, 4, 5])
            if not blade_opts:
                blade_opts = [3, 4, 5]
                st.session_state.blade_options = blade_opts

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

        # Abaqus controls in sidebar
        with st.expander("Abaqus", expanded=True):
            st.markdown("**Load definition**")
            st.number_input("Load magnitude (N)", min_value=0.0, step=50.0, key="abaqus_load_mag")
            st.slider("Force location u along TIP Curve2 TOP (0 to 1)", min_value=0.0, max_value=1.0, step=0.01, key="abaqus_u")
            st.checkbox("Flip normal direction", key="abaqus_flip_normal")

            st.info(
                "The generated Abaqus script applies the load on **TIP Curve2 TOP** by:\n"
                "- finding the correct tip edge using the (x,z) hint from this app,\n"
                "- partitioning that edge by u,\n"
                "- applying the load at the closest node to the partition vertex.\n"
                "If any stage fails, the script stops (no fallback)."
            )

            st.markdown("---")
            st.markdown("**Material**")
            st.text_input("Material name", key="abaqus_material_name")
            st.number_input("Young's modulus E (Pa)", min_value=0.0, step=1e9, format="%.6g", key="abaqus_E")
            st.number_input("Poisson's ratio ν", min_value=0.0, max_value=0.499, step=0.01, format="%.4f", key="abaqus_nu")

            st.markdown("---")
            st.markdown("**Model / part / instance / mesh / job**")
            st.text_input("Abaqus model name (mdb.models key)", key="abaqus_model_name")
            st.text_input("Part name (imported blade part)", key="abaqus_part_name")
            st.text_input("Instance name (created if needed)", key="abaqus_instance_name")
            st.number_input("Global mesh seed size (model units)", min_value=0.0, step=0.1, format="%.6g", key="abaqus_mesh_seed")
            st.text_input("Step name", key="abaqus_step_name")
            st.text_input("Job name", key="abaqus_job_name")

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

            **Abaqus**
            - Script is **post-import**: blade part already exists in CAE.
            - It identifies the **TIP Curve2 TOP edge**, partitions it at **u**, meshes, applies root BC, and applies the load at the partition location.
            - The Abaqus script prints stages to the terminal and stops immediately if any stage fails (no fallback).
            """
        )

    # =========================================================
    # Compute geometry from current session_state
    # =========================================================

    num_blades = int(st.session_state.num_blades)
    blade_length_m, blade_thickness_m = blade_length_and_thickness_from_table(num_blades, variant_table)

    scale_x = float(st.session_state.scale_x)
    scale_z = float(st.session_state.scale_z)
    twist_total_deg = float(st.session_state.twist_total_deg)
    twist_sign = float(st.session_state.twist_sign)

    twist_deg_per_m = twist_total_deg / blade_length_m if blade_length_m > 0 else 0.0

    root_c1_cubic = np.array(
        [
            [st.session_state.c1_p0_x, st.session_state.c1_p0_z],
            [st.session_state.c1_p1_x, st.session_state.c1_p1_z],
            [st.session_state.c1_p2_x, st.session_state.c1_p2_z],
            [st.session_state.c1_p3_x, st.session_state.c1_p3_z],
        ],
        dtype=float,
    )

    root_c2_cubic = compute_curve2_cubic_c2(root_c1_cubic, trailing_edge=(0.0, 0.0))
    c0_res, c1_res, c2_res = continuity_residuals_cubic(root_c1_cubic, root_c2_cubic)

    root_all_cps_top = np.vstack([root_c1_cubic, root_c2_cubic])
    root_all_cps_bottom = root_all_cps_top.copy()
    root_all_cps_bottom[:, 1] *= -1.0

    tip_all_top_2d, tip_all_top_3d = transform_tip_from_root(
        root_all_cps_top, scale_x=scale_x, scale_z=scale_z,
        twist_deg_per_m=twist_deg_per_m, blade_length=blade_length_m,
        twist_sign=twist_sign,
    )
    tip_all_bottom_2d, _ = transform_tip_from_root(
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

    x_range, z_range = compute_axis_ranges(
        root_c1_cubic, root_c2_cubic,
        tip_c1_top_2d, tip_c2_top_2d,
        tip_c1_bottom_2d, tip_c2_bottom_2d,
    )

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

    x_all_root = np.concatenate([root_c1_cubic[:, 0], root_c2_cubic[:, 0]])
    chord_length = float(np.max(x_all_root) - 0.0)

    cfd_x_min = -CFD_X_UPSTREAM_CHORDS * chord_length
    cfd_x_max = CFD_X_DOWNSTREAM_CHORDS * chord_length
    cfd_y_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_y_max = +CFD_RADIAL_CHORDS * chord_length
    cfd_z_min = -CFD_RADIAL_CHORDS * chord_length
    cfd_z_max = +CFD_RADIAL_CHORDS * chord_length

    # --- Abaqus load definition (from TIP curve2-top only) ---
    load_u = float(st.session_state.abaqus_u)
    load_mag = float(st.session_state.abaqus_load_mag)
    flip_normal = bool(st.session_state.abaqus_flip_normal)

    # Use tip curve2-top 2D as the curve we are targeting in the Abaqus script
    pt2d, tan2d = bezier_point_and_tangent(tip_c2_top_2d, load_u)
    x_u, z_u = float(pt2d[0]), float(pt2d[1])
    dxdu, dzdu = float(tan2d[0]), float(tan2d[1])

    # Build a unit normal in XZ plane from tangent
    tangent_3d = np.array([dxdu, 0.0, dzdu], dtype=float)
    normal_3d = np.array([-tangent_3d[2], 0.0, tangent_3d[0]], dtype=float)
    norm_mag = float(np.linalg.norm(normal_3d))
    normal_unit_3d = normal_3d / norm_mag if norm_mag > 0 else np.array([0.0, 0.0, 0.0])

    # Force components
    if flip_normal:
        normal_unit_3d = -normal_unit_3d

    fx = load_mag * float(normal_unit_3d[0])
    fy = load_mag * float(normal_unit_3d[1])
    fz = load_mag * float(normal_unit_3d[2])

    # 2D force direction in X–Z plane (unit, for arrow)
    dir_xz = np.array([normal_unit_3d[0], normal_unit_3d[2]], dtype=float)
    dir_xz_mag = float(np.linalg.norm(dir_xz))
    dir_xz_unit = (dir_xz / dir_xz_mag) if dir_xz_mag > 0 else np.array([0.0, 0.0])

    arrow_len_2d = 0.15 * chord_length if chord_length > 0 else 0.15

    # Load point shown in plots (this is a "target hint", not the actual Abaqus selection coordinate)
    load_point_3d_preview = np.array([x_u, blade_length_m, z_u], dtype=float)

    # =========================================================
    # Outputs / exports
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

    # Design table (single configuration)
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

        "Join_C0_residual": [c0_res],
        "Join_C1_residual": [c1_res],
        "Join_C2_residual": [c2_res],

        # Abaqus intent (tip curve2-top)
        "Abaqus_Load_u": [load_u],
        "Abaqus_LoadMag": [load_mag],
        "LoadHint_X": [x_u],
        "LoadHint_Z": [z_u],
        "LoadDir_X": [float(normal_unit_3d[0])],
        "LoadDir_Y": [float(normal_unit_3d[1])],
        "LoadDir_Z": [float(normal_unit_3d[2])],
        "Force_X": [float(fx)],
        "Force_Y": [float(fy)],
        "Force_Z": [float(fz)],

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
        pd.concat([root_c1_df_out, root_c2_df_out], ignore_index=True).to_excel(writer, sheet_name="Blade_Root_CPs", index=False)
        pd.concat([tip_c1_df_out, tip_c2_df_out], ignore_index=True).to_excel(writer, sheet_name="Blade_Tip_CPs", index=False)
        hub_df_out_for_design.to_excel(writer, sheet_name="Hub_CPs", index=False)
        variant_table.to_excel(writer, sheet_name="BladeVariants", index=False)
    design_table_bytes = buf_dt.getvalue()

    # Config template
    template_bytes = read_template_bytes()
    if template_bytes is None:
        buf_cfg = BytesIO()
        with pd.ExcelWriter(buf_cfg, engine="xlsxwriter") as writer:
            default_variant_table().to_excel(writer, sheet_name="Table1_BladeVariants", index=False)
            pd.DataFrame({
                "Name": [
                    "NumBlades", "ScaleX", "ScaleZ", "TwistTotalDeg", "TwistSign",
                    "AbaqusLoadMag", "AbaqusU",
                    "AbaqusMaterialName", "AbaqusE", "AbaqusNu", "AbaqusMeshSeed",
                    "AbaqusModelName", "AbaqusPartName", "AbaqusInstanceName", "AbaqusStepName", "AbaqusJobName",
                ],
                "Value": [
                    3, 0.8, 0.8, 15.0, +1,
                    1000.0, 0.5,
                    "BladeMaterial", 70e9, 0.33, 0.50,
                    "Model-1", "Model-1", "Blade-1", "Step-1", "BladeJob",
                ]
            }).to_excel(writer, sheet_name="Parameters", index=False)
            pd.DataFrame({
                "Section": ["Root"]*4, "Curve": [1]*4, "PointIndex": [0,1,2,3],
                "X": [1.0, 1.0, 0.8, 0.6], "Z": [0.0, 0.06, 0.08, 0.075]
            }).to_excel(writer, sheet_name="BezierControlPoints", index=False)
            pd.DataFrame({
                "PointIndex": [0,1,2,3],
                "X": [-0.4, 0.3, 13.0, 2.0], "Z": [1.0, 1.0, 0.8, 0.0]
            }).to_excel(writer, sheet_name="HubControlPoints", index=False)
        template_bytes = buf_cfg.getvalue()

    # ---- Abaqus script generation (NEW robust method) ----
    script_lines = generate_abaqus_post_import_script_tip_curve2_partition_u(
        model_name=str(st.session_state.abaqus_model_name),
        part_name=str(st.session_state.abaqus_part_name),
        instance_name=str(st.session_state.abaqus_instance_name),
        material_name=str(st.session_state.abaqus_material_name),
        E=float(st.session_state.abaqus_E),
        nu=float(st.session_state.abaqus_nu),
        shell_thickness=float(blade_thickness_m),
        mesh_seed=float(st.session_state.abaqus_mesh_seed),
        step_name=str(st.session_state.abaqus_step_name),
        job_name=str(st.session_state.abaqus_job_name),
        load_u=float(load_u),
        load_mag=float(load_mag),
        load_point_hint_xz=(float(x_u), float(z_u)),
        force_dir_unit_xyz=(float(normal_unit_3d[0]), float(normal_unit_3d[1]), float(normal_unit_3d[2])),
        flip_dir=False,  # already applied above
    )
    script_bytes = script_lines.encode("utf-8")

    # Downloads expander
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
                    "📥 Download Abaqus Python script (.py)",
                    data=script_bytes,
                    file_name="abaqus_post_import_tip_curve2_partition_u.py",
                    mime="text/x-python",
                )
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
                st.download_button(
                    "Download CFD domain parameters (CSV)",
                    data=cfd_box_df.to_csv(index=False).encode("utf-8"),
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

    # Force summary table (preview intent)
    df_force = pd.DataFrame(
        {
            "Quantity": [
                "Target hint X (from tip C2 top) (m)",
                "Target hint Y (display only) (m)",
                "Target hint Z (from tip C2 top) (m)",
                "Unit direction X",
                "Unit direction Y",
                "Unit direction Z",
                "Force X (N)",
                "Force Y (N)",
                "Force Z (N)",
                "Force magnitude check (N)",
            ],
            "Value": [
                float(load_point_3d_preview[0]),
                float(load_point_3d_preview[1]),
                float(load_point_3d_preview[2]),
                float(normal_unit_3d[0]),
                float(normal_unit_3d[1]),
                float(normal_unit_3d[2]),
                float(fx),
                float(fy),
                float(fz),
                float(np.sqrt(fx*fx + fy*fy + fz*fz)),
            ],
        }
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
            show_tip_c2_u_labels=True,
            force_overlay_enabled=True,
            force_load_xz=(float(load_point_3d_preview[0]), float(load_point_3d_preview[2])),
            force_dir_xz_unit=(float(dir_xz_unit[0]), float(dir_xz_unit[1])),
            force_arrow_len=float(arrow_len_2d),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Continuity check at Curve 1 → Curve 2 join (Root top)")
        st.table(pd.DataFrame({"Residual": ["C0", "C1", "C2"], "Value": [c0_res, c1_res, c2_res]}))

        st.subheader("Abaqus load intent (preview)")
        st.table(df_force)

        st.subheader("Control points for manufacturing – Blade")
        st.markdown("**Root – cubic control points (top half, Y = 0)**")
        st.dataframe(pd.concat([root_c1_df_out, root_c2_df_out], ignore_index=True))
        st.markdown("**Tip – cubic control points (top half, Y = Blade length)**")
        st.dataframe(pd.concat([tip_c1_df_out, tip_c2_df_out], ignore_index=True))

    # ---------- 3D preview tab ----------
    with tab_3d:
        st.header("3D preview – Blade")
        fig3d = make_blade_3d_plot(root_c1_cubic, root_c2_cubic, tip_c1_top_2d, tip_c2_top_2d, blade_length_m)
        fig3d = add_force_to_3d_fig(fig3d, load_point_3d_preview, normal_unit_3d, chord_length, label="Force direction (preview)")
        st.plotly_chart(fig3d, use_container_width=True)

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
        st.subheader("Abaqus script preview (post-import, tip curve2 partition-by-u)")
        st.code(script_lines, language="python")


if __name__ == "__main__":
    main()
