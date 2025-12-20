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
    st.session_state.abaqus_u = 0.50
    st.session_state.abaqus_flip_normal = False

    # Abaqus model database defaults
    st.session_state.abaqus_model_name = "Model-1"
    st.session_state.abaqus_part_name = "Blade"
    st.session_state.abaqus_instance_name = "Blade-1"
    st.session_state.abaqus_material_name = "BladeMaterial"
    st.session_state.abaqus_E = 70e9
    st.session_state.abaqus_nu = 0.33
    st.session_state.abaqus_mesh_seed = 0.25
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
        if "AbaqusFlipNormal" in params:
            try:
                st.session_state.abaqus_flip_normal = bool(int(float(params["AbaqusFlipNormal"])))
            except Exception:
                warnings.append("AbaqusFlipNormal invalid; ignored.")

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

    return applied, warnings, variant_table


# =========================================================
# Abaqus script generator - USES YOUR PROVIDED ABAQUS.PY TEMPLATE
# =========================================================

def generate_abaqus_script_from_your_template(
    model_name: str,
    part_name: str,
    instance_name: str,
    material_name: str,
    step_name: str,
    job_name: str,
    E: float,
    nu: float,
    shell_t: float,
    mesh_seed: float,
    load_u: float,
    load_point_xyz: Tuple[float, float, float],
    force_vec_xyz: Tuple[float, float, float],
) -> str:
    # Guard: avoid u=0 or u=1 which can break PartitionEdgeByParam
    u = float(load_u)
    u = min(max(u, 1e-6), 1.0 - 1e-6)

    lx, ly, lz = [float(v) for v in load_point_xyz]
    fx, fy, fz = [float(v) for v in force_vec_xyz]

    return f"""# -*- coding: utf-8 -*-
# Abaqus/CAE script (POST-IMPORT) generated by Streamlit
# Assumes your blade part already exists in the model database.

from abaqus import mdb
from abaqusConstants import *
import regionToolset
import mesh
import math

# =========================================================
# USER INPUTS (match your model tree)
# =========================================================
MODEL_NAME    = r"{model_name}"
PART_NAME     = r"{part_name}"
INSTANCE_NAME = r"{instance_name}"

MATERIAL_NAME = r"{material_name}"
STEP_NAME     = r"{step_name}"
JOB_NAME      = r"{job_name}"

# Material / section
E       = {E:.16g}
NU      = {nu:.16g}
SHELL_T = {shell_t:.16g}

# Mesh
MESH_SEED = {mesh_seed:.16g}

# Load definition (from Streamlit)
LOAD_U = {u:.16g}
LOAD_POINT = ({lx:.16g}, {ly:.16g}, {lz:.16g})
FORCE_VEC  = ({fx:.16g}, {fy:.16g}, {fz:.16g})

# =========================================================
# UNIT HANDLING
# =========================================================
AUTO_UNIT_SCALE = True
UNIT_SCALE = 1.0
# If AUTO_UNIT_SCALE=True, UNIT_SCALE will be inferred from geometry vs LOAD_POINT.y

# =========================================================
# Logging
# =========================================================
def log(msg):
    print(">>> " + str(msg))

def die(msg):
    raise RuntimeError(str(msg))

# =========================================================
# Basic getters
# =========================================================
def get_model():
    if MODEL_NAME not in mdb.models:
        die("Model '%s' not found. Available models: %s" % (MODEL_NAME, list(mdb.models.keys())))
    return mdb.models[MODEL_NAME]

def get_part(model):
    if PART_NAME not in model.parts:
        die("Part '%s' not found. Available parts: %s" % (PART_NAME, list(model.parts.keys())))
    return model.parts[PART_NAME]

def get_or_create_instance(model, part_obj):
    a = model.rootAssembly
    if INSTANCE_NAME in a.instances:
        log("Using existing instance: %s" % INSTANCE_NAME)
        return a.instances[INSTANCE_NAME]

    if len(a.instances) > 0:
        inst0 = a.instances.values()[0]
        log("WARNING: Instance '%s' not found; reusing first existing instance: %s" % (INSTANCE_NAME, inst0.name))
        return inst0

    log("No instances found. Creating instance %s." % INSTANCE_NAME)
    a.DatumCsysByDefault(CARTESIAN)
    inst = a.Instance(name=INSTANCE_NAME, part=part_obj, dependent=ON)
    return inst

# =========================================================
# Bounding box WITHOUT part.getBoundingBox()
# =========================================================
def bbox_from_part_geometry(part_obj):
    \"\"\"
    Compute an approximate bounding box by sampling geometry points.
    Works in Abaqus versions where Part.getBoundingBox() does not exist.
    \"\"\"
    pts = []

    # Prefer vertices
    try:
        verts = part_obj.vertices
        for v in verts:
            p = v.pointOn[0]
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    except Exception as ex:
        log("WARNING: Could not read vertices for bbox: %s" % ex)

    # Fallback: edge sample points
    if len(pts) == 0:
        try:
            for e in part_obj.edges:
                p = e.pointOn[0]
                pts.append((float(p[0]), float(p[1]), float(p[2])))
        except Exception as ex:
            die("Failed to compute bbox from geometry points (no vertices/edges usable): %s" % ex)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]

    low  = (min(xs), min(ys), min(zs))
    high = (max(xs), max(ys), max(zs))
    return low, high

def detect_unit_scale_from_geometry(part_obj, load_point):
    low, high = bbox_from_part_geometry(part_obj)
    y_min, y_max = float(low[1]), float(high[1])
    tip_y = y_max
    root_y = y_min

    lp_y = float(load_point[1])
    if not AUTO_UNIT_SCALE:
        return 1.0, root_y, tip_y, low, high

    if abs(lp_y) < 1e-12:
        return 1.0, root_y, tip_y, low, high

    ratio = abs(tip_y / lp_y)

    if 200.0 < ratio < 5000.0:
        scale = ratio
        log("AUTO_UNIT_SCALE: Detected geometry/load mismatch.")
        log("  tip_y ≈ %.6g, LOAD_POINT.y = %.6g, ratio ≈ %.6g -> using UNIT_SCALE=%.6g" % (tip_y, lp_y, ratio, scale))
        return scale, root_y, tip_y, low, high

    return 1.0, root_y, tip_y, low, high

# =========================================================
# Material and section
# =========================================================
def ensure_material_and_section(model, part_obj, shell_t):
    log("Ensuring material and shell section...")
    if MATERIAL_NAME not in model.materials:
        mat = model.Material(name=MATERIAL_NAME)
        mat.Elastic(table=((E, NU),))
        log("Created material: %s" % MATERIAL_NAME)

    sec_name = "BladeShellSection"
    if sec_name not in model.sections:
        model.HomogeneousShellSection(
            name=sec_name,
            material=MATERIAL_NAME,
            thicknessType=UNIFORM,
            thickness=shell_t,
            poissonDefinition=DEFAULT,
            thicknessModulus=None,
            temperature=GRADIENT,
            useDensity=OFF,
            integrationRule=SIMPSON,
            numIntPts=5,
        )
        log("Created section: %s (thickness=%.6g)" % (sec_name, shell_t))

    faces = part_obj.faces
    if len(faces) == 0:
        die("Part has no faces. Are you sure the imported blade is a shell/surface part?")

    region = regionToolset.Region(faces=faces)
    part_obj.SectionAssignment(
        region=region,
        sectionName=sec_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField="",
        thicknessAssignment=FROM_SECTION,
    )
    log("Assigned section to all faces.")

def ensure_step(model):
    log("Ensuring analysis step...")
    if STEP_NAME in model.steps:
        log("Step exists: %s" % STEP_NAME)
        return
    model.StaticStep(name=STEP_NAME, previous="Initial")
    log("Created step: %s" % STEP_NAME)

# =========================================================
# Edge selection: Curve2 top at tip (STRICT)
# =========================================================
def edge_avg_point(edge_obj):
    pts = edge_obj.pointOn
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    zs = [float(p[2]) for p in pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))

def dist3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def find_tip_top_edge_near_loadpoint(inst, tip_y, tol_y, load_point):
    log("Finding tip top edge nearest LOAD_POINT on tip plane...")

    tip_edges = inst.edges.getByBoundingBox(
        xMin=-1e20, xMax=1e20,
        yMin=tip_y - tol_y, yMax=tip_y + tol_y,
        zMin=-1e20, zMax=1e20
    )
    log("Tip-plane edges found: %d" % len(tip_edges))
    if len(tip_edges) == 0:
        die("No edges found near tip plane y=%.6g (tol=%.6g). Check span direction and geometry." % (tip_y, tol_y))

    top_edges = []
    for e in tip_edges:
        p = edge_avg_point(e)
        if p[2] >= 0.0:
            top_edges.append(e)
    log("Top tip edges (z>=0) candidates: %d" % len(top_edges))
    if len(top_edges) == 0:
        die("Found tip-plane edges, but none qualify as TOP (z>=0). Check coordinate system / sign convention.")

    best = None
    best_d = None
    best_p = None
    for e in top_edges:
        p = edge_avg_point(e)
        d = dist3(p, load_point)
        if (best is None) or (d < best_d):
            best = e
            best_d = d
            best_p = p

    if best is None:
        die("Failed to pick a best tip top edge (unexpected).")

    log("Selected tip-top edge: index=%s, avgPt=(%.6g, %.6g, %.6g), dist_to_LOAD_POINT≈%.6g"
        % (str(best.index), best_p[0], best_p[1], best_p[2], best_d))
    return best

# =========================================================
# Partition edge by parameter u (STRICT)
# =========================================================
def partition_edge_by_u(part_obj, inst_edge, u_param):
    log("Partitioning selected tip edge at u=%.4f ..." % float(u_param))

    try:
        part_edge = part_obj.edges[inst_edge.index]
    except Exception as ex:
        die("Could not map instance edge to part edge using index=%s: %s" % (str(inst_edge.index), ex))

    try:
        part_obj.PartitionEdgeByParam(edges=(part_edge,), parameter=float(u_param))
    except Exception as ex:
        die("PartitionEdgeByParam failed. (u=%.4f) Error: %s" % (float(u_param), ex))

    log("PartitionEdgeByParam succeeded at u=%.4f" % float(u_param))

# =========================================================
# Root BC
# =========================================================
def apply_root_pinned_bc(model, inst, root_y, tol_y):
    log("Applying pinned BC at root (y≈%.6g, tol=%.6g)..." % (root_y, tol_y))

    edges = inst.edges.getByBoundingBox(
        xMin=-1e20, xMax=1e20,
        yMin=root_y - tol_y, yMax=root_y + tol_y,
        zMin=-1e20, zMax=1e20
    )
    log("Root-plane edges found: %d" % len(edges))
    if len(edges) == 0:
        die("No root edges found near y=%.6g (tol=%.6g). Check span direction." % (root_y, tol_y))

    a = model.rootAssembly
    set_name = "SET_ROOT_EDGES"
    if set_name in a.sets:
        del a.sets[set_name]
    a.Set(name=set_name, edges=edges)

    region = a.sets[set_name]
    bc_name = "BC_ROOT_PINNED"
    if bc_name in model.boundaryConditions:
        del model.boundaryConditions[bc_name]

    model.DisplacementBC(
        name=bc_name,
        createStepName="Initial",
        region=region,
        u1=0.0, u2=0.0, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None
    )
    log("Pinned BC applied on root edges.")

# =========================================================
# Meshing
# =========================================================
def mesh_part(part_obj, mesh_seed):
    log("Meshing part (seed=%.6g)..." % float(mesh_seed))

    try:
        part_obj.setMeshControls(regions=part_obj.faces, elemShape=QUAD, technique=FREE)
    except Exception:
        pass

    part_obj.seedPart(size=float(mesh_seed), deviationFactor=0.1, minSizeFactor=0.1)

    elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=S3,  elemLibrary=STANDARD)
    part_obj.setElementType(regions=(part_obj.faces,), elemTypes=(elemType1, elemType2))

    part_obj.generateMesh()
    log("Mesh generated.")

# =========================================================
# Closest node robust handling
# =========================================================
def closest_node(nodes_obj, xyz):
    res = nodes_obj.getClosest(coordinates=xyz)

    if hasattr(res, "label"):
        return res

    if isinstance(res, (list, tuple)):
        if len(res) == 0:
            return None
        first = res[0]
        if isinstance(first, (list, tuple)) and len(first) >= 1 and hasattr(first[0], "label"):
            return first[0]
        if hasattr(first, "label"):
            return first

    return None

def apply_concentrated_force_at_loadpoint(model, inst, load_point, force_vec):
    log("Applying concentrated force at closest node to LOAD_POINT...")

    nodes = inst.nodes
    if len(nodes) == 0:
        die("No mesh nodes on instance. Meshing failed or not generated.")

    node_obj = closest_node(nodes, load_point)
    if node_obj is None:
        die("Could not find closest node to LOAD_POINT=%s. Check mesh and coordinates." % str(load_point))

    log("Closest node label=%d at approx LOAD_POINT=%s" % (int(node_obj.label), str(load_point)))

    a = model.rootAssembly
    set_name = "SET_LOAD_NODE"
    if set_name in a.sets:
        del a.sets[set_name]
    a.Set(name=set_name, nodes=inst.nodes.sequenceFromLabels((node_obj.label,)))

    load_name = "LOAD_F"
    if load_name in model.loads:
        del model.loads[load_name]

    model.ConcentratedForce(
        name=load_name,
        createStepName=STEP_NAME,
        region=a.sets[set_name],
        cf1=float(force_vec[0]),
        cf2=float(force_vec[1]),
        cf3=float(force_vec[2]),
        distributionType=UNIFORM,
        field="",
        localCsys=None
    )
    log("Concentrated force applied: (%.6g, %.6g, %.6g) N" % (force_vec[0], force_vec[1], force_vec[2]))

# =========================================================
# Job
# =========================================================
def make_job_and_run():
    log("Creating and running job: %s" % JOB_NAME)
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
    log("Job completed.")

# =========================================================
# MAIN
# =========================================================
def main():
    log("===== START POST-IMPORT BLADE SCRIPT =====")

    model = get_model()
    part_obj = get_part(model)
    inst = get_or_create_instance(model, part_obj)

    log("Stage: detect geometry extents / unit scale")
    inferred_scale, root_y_raw, tip_y_raw, bb_low, bb_high = detect_unit_scale_from_geometry(part_obj, LOAD_POINT)

    global UNIT_SCALE
    UNIT_SCALE = inferred_scale if AUTO_UNIT_SCALE else UNIT_SCALE

    log("BBox (geometry sample): low=%s high=%s" % (str(bb_low), str(bb_high)))
    log("Detected root_y=%.6g, tip_y=%.6g" % (float(root_y_raw), float(tip_y_raw)))

    shell_t_scaled = float(SHELL_T) * float(UNIT_SCALE)
    mesh_seed_scaled = float(MESH_SEED) * float(UNIT_SCALE)
    load_point_scaled = (float(LOAD_POINT[0]) * float(UNIT_SCALE),
                         float(LOAD_POINT[1]) * float(UNIT_SCALE),
                         float(LOAD_POINT[2]) * float(UNIT_SCALE))

    log("UNIT_SCALE=%.6g" % float(UNIT_SCALE))
    log("Scaled: SHELL_T=%.6g, MESH_SEED=%.6g, LOAD_POINT=%s"
        % (shell_t_scaled, mesh_seed_scaled, str(load_point_scaled)))

    span = float(tip_y_raw - root_y_raw)
    tol_y = max(1e-6, 1e-4 * abs(span) if abs(span) > 0 else 1e-6)
    log("tol_y=%.6g (based on span=%.6g)" % (tol_y, span))

    log("Stage: material + section")
    ensure_material_and_section(model, part_obj, shell_t_scaled)

    log("Stage: step")
    ensure_step(model)

    log("Stage: identify tip top edge near LOAD_POINT")
    curve2_tip_edge = find_tip_top_edge_near_loadpoint(inst, tip_y_raw, tol_y, load_point_scaled)

    log("Stage: partition selected edge at u")
    partition_edge_by_u(part_obj, curve2_tip_edge, LOAD_U)

    log("Stage: regenerate assembly (post-partition)")
    model.rootAssembly.regenerate()

    log("Stage: mesh")
    mesh_part(part_obj, mesh_seed_scaled)

    log("Stage: regenerate assembly (post-mesh)")
    model.rootAssembly.regenerate()

    log("Stage: root BC")
    apply_root_pinned_bc(model, inst, root_y_raw, tol_y)

    log("Stage: apply load")
    apply_concentrated_force_at_loadpoint(model, inst, load_point_scaled, FORCE_VEC)

    log("Stage: run job")
    make_job_and_run()

    log("===== END POST-IMPORT BLADE SCRIPT =====")

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

            # IMPORTANT: keep u away from exactly 0 or 1 to avoid partition edge issues
            st.slider("Force location u along TIP Curve2 TOP", min_value=0.01, max_value=0.99, step=0.01, key="abaqus_u")

            st.checkbox("Flip normal direction", key="abaqus_flip_normal")

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

    tab_overview, tab_blade, tab_3d, tab_hub, tab_cfd, tab_abaqus = st.tabs(
        ["Overview", "Blade geometry", "3D preview", "Hub geometry", "CFD domain", "Abaqus config"]
    )

    with tab_overview:
        st.header("Overview")
        if config_applied:
            st.success("Config applied (as new defaults). You can now tweak values in the sidebar without them reverting.")
        if config_warnings:
            for w in config_warnings:
                st.info(w)

        st.markdown(
            """
            **Abaqus script generation**
            - This app generates an Abaqus script using your **known-good Abaqus.py template**
            - The only changes are injected UI values (model/part names, material, thickness, mesh seed, load u, LOAD_POINT, FORCE_VEC)
            - The script avoids `part.getBoundingBox()` so it works on older Abaqus versions.
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

    # --- Abaqus load definition (Tip curve2-top) ---
    load_u = float(st.session_state.abaqus_u)
    load_mag = float(st.session_state.abaqus_load_mag)
    flip_normal = bool(st.session_state.abaqus_flip_normal)

    pt2d, tan2d = bezier_point_and_tangent(tip_c2_top_2d, load_u)
    x_u, z_u = float(pt2d[0]), float(pt2d[1])
    dxdu, dzdu = float(tan2d[0]), float(tan2d[1])

    tangent_3d = np.array([dxdu, 0.0, dzdu], dtype=float)
    normal_3d = np.array([-tangent_3d[2], 0.0, tangent_3d[0]], dtype=float)
    norm_mag = float(np.linalg.norm(normal_3d))
    normal_unit_3d = normal_3d / norm_mag if norm_mag > 0 else np.array([0.0, 0.0, 0.0])

    if flip_normal:
        normal_unit_3d = -normal_unit_3d

    force_vec = (
        load_mag * float(normal_unit_3d[0]),
        load_mag * float(normal_unit_3d[1]),
        load_mag * float(normal_unit_3d[2]),
    )

    # load point in the blade coords (used as LOAD_POINT in the Abaqus template)
    load_point_3d = (x_u, float(blade_length_m), z_u)

    # 2D arrow direction
    dir_xz = np.array([normal_unit_3d[0], normal_unit_3d[2]], dtype=float)
    dir_xz_mag = float(np.linalg.norm(dir_xz))
    dir_xz_unit = (dir_xz / dir_xz_mag) if dir_xz_mag > 0 else np.array([0.0, 0.0])
    arrow_len_2d = 0.15 * chord_length if chord_length > 0 else 0.15

    # =========================================================
    # Exports
    # =========================================================

    # Design table (single configuration)
    dt_df = pd.DataFrame(
        {
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
            "Abaqus_Load_u": [load_u],
            "Abaqus_LoadMag": [load_mag],
            "LOAD_POINT_X": [load_point_3d[0]],
            "LOAD_POINT_Y": [load_point_3d[1]],
            "LOAD_POINT_Z": [load_point_3d[2]],
            "FORCE_X": [force_vec[0]],
            "FORCE_Y": [force_vec[1]],
            "FORCE_Z": [force_vec[2]],
            "CFD_Xmin": [cfd_x_min],
            "CFD_Xmax": [cfd_x_max],
            "CFD_Ymin": [cfd_y_min],
            "CFD_Ymax": [cfd_y_max],
            "CFD_Zmin": [cfd_z_min],
            "CFD_Zmax": [cfd_z_max],
            "CFD_Chord": [chord_length],
        }
    )

    buf_dt = BytesIO()
    with pd.ExcelWriter(buf_dt, engine="xlsxwriter") as writer:
        dt_df.to_excel(writer, sheet_name="DesignTable", index=False)
        variant_table.to_excel(writer, sheet_name="BladeVariants", index=False)
    design_table_bytes = buf_dt.getvalue()

    # Config template (fixes the earlier bug where PartName mistakenly became "Model-1")
    template_bytes = read_template_bytes()
    if template_bytes is None:
        buf_cfg = BytesIO()
        with pd.ExcelWriter(buf_cfg, engine="xlsxwriter") as writer:
            default_variant_table().to_excel(writer, sheet_name="Table1_BladeVariants", index=False)
            pd.DataFrame({
                "Name": [
                    "NumBlades", "ScaleX", "ScaleZ", "TwistTotalDeg", "TwistSign",
                    "AbaqusLoadMag", "AbaqusU", "AbaqusFlipNormal",
                    "AbaqusMaterialName", "AbaqusE", "AbaqusNu", "AbaqusMeshSeed",
                    "AbaqusModelName", "AbaqusPartName", "AbaqusInstanceName", "AbaqusStepName", "AbaqusJobName",
                ],
                "Value": [
                    3, 0.8, 0.8, 15.0, +1,
                    1000.0, 0.5, 0,
                    "BladeMaterial", 70e9, 0.33, 0.25,
                    "Model-1", "Blade", "Blade-1", "Step-1", "BladeJob",
                ]
            }).to_excel(writer, sheet_name="Parameters", index=False)
        template_bytes = buf_cfg.getvalue()

    # Generate Abaqus script from YOUR template
    script_text = generate_abaqus_script_from_your_template(
        model_name=str(st.session_state.abaqus_model_name),
        part_name=str(st.session_state.abaqus_part_name),
        instance_name=str(st.session_state.abaqus_instance_name),
        material_name=str(st.session_state.abaqus_material_name),
        step_name=str(st.session_state.abaqus_step_name),
        job_name=str(st.session_state.abaqus_job_name),
        E=float(st.session_state.abaqus_E),
        nu=float(st.session_state.abaqus_nu),
        shell_t=float(blade_thickness_m),
        mesh_seed=float(st.session_state.abaqus_mesh_seed),
        load_u=float(load_u),
        load_point_xyz=load_point_3d,
        force_vec_xyz=force_vec,
    )
    script_bytes = script_text.encode("utf-8")

    with st.sidebar:
        with downloads_placeholder.container():
            with st.expander("Downloads", expanded=False):
                st.download_button(
                    "📥 Download CATIA design table (Excel)",
                    data=design_table_bytes,
                    file_name="blade_design_table.xlsx",
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
                    file_name="abaqus_post_import_from_template.py",
                    mime="text/x-python",
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
            force_load_xz=(float(load_point_3d[0]), float(load_point_3d[2])),
            force_dir_xz_unit=(float(dir_xz_unit[0]), float(dir_xz_unit[1])),
            force_arrow_len=float(arrow_len_2d),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Continuity check at Curve 1 → Curve 2 join (Root top)")
        st.table(pd.DataFrame({"Residual": ["C0", "C1", "C2"], "Value": [c0_res, c1_res, c2_res]}))

        st.subheader("Abaqus load values that will be injected")
        st.table(pd.DataFrame({
            "Name": ["LOAD_U", "LOAD_POINT (x,y,z)", "FORCE_VEC (Fx,Fy,Fz)"],
            "Value": [load_u, str(load_point_3d), str(force_vec)]
        }))

    # ---------- 3D preview tab ----------
    with tab_3d:
        st.header("3D preview – Blade")
        fig3d = make_blade_3d_plot(root_c1_cubic, root_c2_cubic, tip_c1_top_2d, tip_c2_top_2d, blade_length_m)
        fig3d = add_force_to_3d_fig(fig3d, np.array(load_point_3d), normal_unit_3d, chord_length, label="Force direction (preview)")
        st.plotly_chart(fig3d, use_container_width=True)

    # ---------- Hub geometry tab ----------
    with tab_hub:
        st.header("Hub geometry")
        st.markdown(f"Hub profile control points source: **{hub_source_label}**")
        st.plotly_chart(make_hub_plot(hub_cp_az, hub_plane_label=hub_plane_label), use_container_width=True)

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
        st.header("Abaqus script preview (generated from your template)")
        st.code(script_text, language="python")


if __name__ == "__main__":
    main()
