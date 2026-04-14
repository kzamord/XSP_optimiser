import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch
import joblib
import os
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="xSP Optimiser",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #1c2b1c; }
    section[data-testid="stSidebar"] { background-color: #162016; border-right: 1px solid #2d6a2d; }
    .stApp, .stMarkdown, p, label { color: #ffffff !important; }
    .stSlider label { color: #ffffff !important; }
    h1 { color: #F2A623 !important; font-family: 'Georgia', serif; }
    h2, h3 { color: #ffffff !important; }
    .stRadio label { color: #ffffff !important; }
    .stSelectbox label { color: #aaaaaa !important; font-style: italic; }
    hr { border-color: #2d6a2d; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
GOAL_X      = 120.0
GOAL_Y      = 40.0
GOAL_W      = 7.32
FINAL_THIRD = 80.0
BOX_ZONE    = 102.0
RESOLUTION  = 35

# ── Load models (cached permanently) ─────────────────────────────────────
@st.cache_resource
def load_models():
    BASE = os.path.dirname(os.path.abspath(__file__))
    fk_model            = joblib.load(os.path.join(BASE, "xsp_model.pkl"))
    corner_model        = joblib.load(os.path.join(BASE, "xsp_corner_model.pkl"))
    fk_teams_dict       = joblib.load(os.path.join(BASE, "fk_opp_profiles.pkl"))
    corner_teams_dict   = joblib.load(os.path.join(BASE, "corner_opp_profiles.pkl"))
    fk_teams_sorted     = joblib.load(os.path.join(BASE, "fk_teams_list.pkl"))
    corner_teams_sorted = joblib.load(os.path.join(BASE, "corner_teams_list.pkl"))
    FK_COLS             = fk_model.get_booster().feature_names
    CORNER_COLS         = corner_model.get_booster().feature_names
    return (fk_model, corner_model, FK_COLS, CORNER_COLS,
            fk_teams_dict, corner_teams_dict,
            fk_teams_sorted, corner_teams_sorted)

with st.spinner("⚽ Loading models..."):
    (fk_model, corner_model, FK_COLS, CORNER_COLS,
     fk_teams_dict, corner_teams_dict,
     fk_teams_sorted, corner_teams_sorted) = load_models()

# ── Grid computation (cached per unique parameter combination) ────────────
@st.cache_data
def generate_grid(mode, wall=0, defenders=6, attackers=7, end_z=1.2,
                  defenders_box=6, attackers_box=7,
                  delivery_val=0, side_val=0, resolution=RESOLUTION):

    # ── Corner scans box zone only — matching training data distribution ──
    if mode == "FK":
        x_vals = np.linspace(80, 118, resolution)
        y_vals = np.linspace(2, 78, resolution)
    else:
        x_vals = np.linspace(100, 120, resolution)
        y_vals = np.linspace(15, 65, resolution)

    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    x_flat = X_grid.ravel()
    y_flat = Y_grid.ravel()

    dx      = GOAL_X - x_flat
    dy      = y_flat - GOAL_Y
    angle   = np.abs(np.degrees(np.arctan2(GOAL_W*dx, dx**2+dy**2-(GOAL_W/2)**2)))
    lateral = np.abs(y_flat - GOAL_Y)
    zxd = (x_flat < BOX_ZONE).astype(int)
    zxb = (x_flat >= BOX_ZONE).astype(int)
    zlw = (y_flat < 16.8).astype(int)
    zlh = ((y_flat >= 16.8) & (y_flat < 29.6)).astype(int)
    zc  = ((y_flat >= 29.6) & (y_flat < 50.4)).astype(int)
    zrh = ((y_flat >= 50.4) & (y_flat < 63.2)).astype(int)
    zrw = (y_flat >= 63.2).astype(int)

    if mode == "FK":
        df = pd.DataFrame({
            "x": x_flat, "y": y_flat,
            "angle_to_goal": angle, "lateral_offset": lateral,
            "shot_end_y": y_flat, "shot_end_z": end_z,
            "is_foot": 1, "is_head": 0,
            "is_first_time": 0, "is_deflected": 0, "under_pressure": 0,
            "ff_n_defenders": defenders,
            "ff_n_attackers": attackers,
            "ff_gk_present": 1,
            "ff_wall_size": wall,
            "minute": 45, "xg": 0.05,
            "zone_x_Deep zone": zxd,
            "zone_y_Left wide": zlw, "zone_y_Left half": zlh,
            "zone_y_Central": zc, "zone_y_Right half": zrh,
            "zone_y_Right wide": zrw,
        })
        preds = fk_model.predict_proba(df[FK_COLS])[:, 1]
        preds[x_flat >= BOX_ZONE] = 0.0

    else:
        is_ins = 1 if delivery_val == 0 else 0
        is_str = 1 if delivery_val == 1 else 0
        is_out = 1 if delivery_val == 2 else 0
        left_c = 1 if side_val == 0 else 0
        df = pd.DataFrame({
            "x": x_flat, "y": y_flat,
            "angle_to_goal": angle, "lateral_offset": lateral,
            "shot_end_y": y_flat, "shot_end_z": end_z,
            "is_head": 1, "is_aerial_won": 0, "is_open_goal": 0,
            "is_first_time": 0, "is_deflected": 0, "under_pressure": 0,
            "ff_n_defenders": defenders_box + 2,
            "ff_n_attackers": attackers_box,
            "ff_defenders_box": defenders_box,
            "is_inswinging": is_ins, "is_outswinging": is_out,
            "is_straight": is_str, "is_left_corner": left_c,
            "minute": 45, "xg": 0.07,
            "zone_x_Box zone": zxb,
            "zone_y_Left wide": zlw, "zone_y_Left half": zlh,
            "zone_y_Central": zc, "zone_y_Right half": zrh,
            "zone_y_Right wide": zrw,
        })
        preds = corner_model.predict_proba(df[CORNER_COLS])[:, 1]

    return x_vals, y_vals, preds.reshape(len(y_vals), len(x_vals))

# ── Zone label helper ─────────────────────────────────────────────────────
def get_zone_label(bx, by):
    depth = "Box zone" if bx >= BOX_ZONE else "Deep zone"
    if   by < 16.8: lane = "Left wide"
    elif by < 29.6: lane = "Left half"
    elif by < 36.3: lane = "Left central"
    elif by < 43.7: lane = "Central"
    elif by < 50.4: lane = "Right central"
    elif by < 63.2: lane = "Right half"
    else:           lane = "Right wide"
    return depth + " - " + lane

# ── Figure builder (cached per unique parameter combination) ──────────────
@st.cache_data
def build_figure(mode, opp_name, wall=0, defenders=6, attackers=7, end_z=1.2,
                 defenders_box=6, attackers_box=7,
                 delivery_val=0, side_val=0):

    x_vals, y_vals, grid = generate_grid(
        mode, wall=wall, defenders=defenders, attackers=attackers, end_z=end_z,
        defenders_box=defenders_box, attackers_box=attackers_box,
        delivery_val=delivery_val, side_val=side_val
    )

    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("#1c2b1c")
    ax = fig.add_axes([0.05, 0.05, 0.82, 0.88])

    pitch = VerticalPitch(
        pitch_type="statsbomb", pitch_color="#2d6a2d",
        line_color="white", stripe=False,
        goal_type="box", linewidth=1.5, half=True
    )
    pitch.draw(ax=ax)
    ax.set_facecolor("#1c2b1c")

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "xsp", ["#1a4a1a","#2d8a2d","#f5c518","#e8531c","#c0392b"]
    )
    norm = mcolors.Normalize(vmin=0, vmax=0.35)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    ax.pcolormesh(Y_grid, X_grid, grid, cmap=cmap, norm=norm,
                  alpha=0.75, zorder=3, shading="gouraud")

    if mode == "FK":
        ax.add_patch(Rectangle(
            (0, BOX_ZONE), 80, 120 - BOX_ZONE,
            facecolor="#1c2b1c", alpha=0.65, zorder=4, linewidth=0
        ))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("xSP score", color="white", fontsize=9)
    cbar.ax.tick_params(colors="white")

    ax.axhline(FINAL_THIRD, color="white", linewidth=1.5, linestyle="--", alpha=0.7, zorder=6)
    ax.axhline(BOX_ZONE, color="white", linewidth=0.9, linestyle="--", alpha=0.5, zorder=6)

    # Main zone lines
    for xl in [16.8, 29.6, 50.4, 63.2]:
        ax.plot([xl,xl],[FINAL_THIRD,120], color="white",
                linewidth=0.7, linestyle="--", alpha=0.4, zorder=6)

    # Near post / far post — orange dotted, box zone only
    for xl in [36.3, 43.7]:
        ax.plot([xl,xl],[BOX_ZONE,120], color="#F2A623",
                linewidth=0.8, linestyle=":", alpha=0.7, zorder=6)

    # Zone labels
    lane_centers = [8.4, 23.2, 33.0, 40.0, 47.0, 56.8, 71.6]
    zone_names   = ["Left wide","Left half","Left central","Central",
                    "Right central","Right half","Right wide"]
    deep_mid = (FINAL_THIRD + BOX_ZONE) / 2
    box_mid  = (BOX_ZONE + 120) / 2
    for xc, name in zip(lane_centers, zone_names):
        ax.text(xc, deep_mid, name, color="white", fontsize=5,
                ha="center", va="center", alpha=0.6, zorder=7)
        ax.text(xc, box_mid, name, color="white", fontsize=5,
                ha="center", va="center", alpha=0.6, zorder=7)
    ax.text(-4, deep_mid, "Deep zone", color="white", fontsize=7,
            ha="center", va="center", alpha=0.8, rotation=90)
    ax.text(-4, box_mid, "Box zone", color="white", fontsize=7,
            ha="center", va="center", alpha=0.8, rotation=90)

    # Best spot
    bi    = np.unravel_index(grid.argmax(), grid.shape)
    bx    = x_vals[bi[1]]
    by    = y_vals[bi[0]]
    bdist = np.sqrt((GOAL_X-bx)**2 + (by-GOAL_Y)**2)
    bzone = get_zone_label(bx, by)

    ax.plot(y_vals[bi[0]], x_vals[bi[1]], "*",
            color="#FFD700", markersize=20,
            markeredgecolor="white", markeredgewidth=1.5, zorder=10)
    ax.text(40, 82,
            f"★ Best xSP: {grid.max()*100:.1f}%  |  {bdist:.1f}m  |  {bzone}",
            color="white", fontsize=8, ha="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="#FFD700",
                      linewidth=1, boxstyle="round,pad=0.4"), zorder=11)
    ax.text(40, FINAL_THIRD - 3,
            "xSP guide:  Low < 5%   |   Moderate 5%–15%   |   High > 15%",
            color="#aaaaaa", fontsize=6, ha="center", va="top",
            style="italic", zorder=6)

    color = "#F2A623" if mode == "FK" else "#4FC3F7"
    label = "Direct Free Kick" if mode == "FK" else "Corner"
    ax.set_title(f"xSP Optimiser — {label}  |  vs {opp_name}",
                 color=color, fontsize=11, fontweight="bold", pad=10)

    return fig, float(grid.max()), float(bdist), bzone

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ xSP Optimiser")
    st.markdown("---")

    mode = st.radio("Mode", ["Free Kick", "Corner"], horizontal=True)
    st.markdown("---")

    if mode == "Free Kick":
        st.markdown("### 🟠 Free Kick Parameters")
        opp = st.selectbox("Opposition team", fk_teams_sorted, index=0)

        if opp == "Average":
            def_wall, def_def, def_att = 5, 5, 7
        elif opp in fk_teams_dict:
            p = fk_teams_dict[opp]
            wall_tmp = p["wall"]
            def_wall = wall_tmp
            def_def  = min(p["defenders"], 10 - wall_tmp)
            def_att  = p["attackers"]
        else:
            def_wall, def_def, def_att = 5, 5, 7

        wall      = st.slider("Wall defenders",   0, 10, def_wall)
        defenders = st.slider("Defenders in box", 0, max(0, 10 - wall), min(def_def, 10 - wall))
        attackers = st.slider("Attackers",        0,  9, def_att)
        end_z     = st.slider("Shot height (m)",  0.1, 2.8, 1.2, step=0.1)

        # Warning if total exceeds 10
        if wall + defenders > 10:
            st.warning(f"⚠️ Wall ({wall}) + Defenders ({defenders}) = {wall+defenders} > 10")
            defenders = 10 - wall

        grid_kwargs = dict(wall=wall, defenders=defenders,
                           attackers=attackers, end_z=end_z)

    else:
        st.markdown("### 🔵 Corner Parameters")
        opp = st.selectbox("Opposition team", corner_teams_sorted, index=0)

        if opp == "Average":
            def_dbox, def_abox = 6, 7
        elif opp in corner_teams_dict:
            p = corner_teams_dict[opp]
            def_dbox, def_abox = p["defenders_box"], p["attackers_box"]
        else:
            def_dbox, def_abox = 6, 7

        defenders_box = st.slider("Defenders in box", 0, 11, def_dbox)
        attackers_box = st.slider("Attackers",        0,  9, def_abox)
        end_z         = st.slider("Shot height (m)",  0.1, 2.8, 1.2, step=0.1)
        delivery      = st.select_slider("Delivery type",
                                         ["Inswinging", "Straight", "Outswinging"])
        side          = st.radio("Corner side", ["Left", "Right"], horizontal=True)

        delivery_val = ["Inswinging", "Straight", "Outswinging"].index(delivery)
        side_val     = 0 if side == "Left" else 1

        grid_kwargs = dict(defenders_box=defenders_box, attackers_box=attackers_box,
                           end_z=end_z, delivery_val=delivery_val, side_val=side_val)

    st.markdown("---")
    run = st.button("🔄 Refresh heatmap", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────
mode_key = "FK" if mode == "Free Kick" else "Corner"

st.markdown("<h1 style='text-align:center'>⚽ xSP Optimiser</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaaaaa'>Select mode · Pick opposition · Adjust sliders · Click Refresh</p>",
            unsafe_allow_html=True)

if run or "last_key" not in st.session_state:
    with st.spinner("Computing xSP heatmap..."):
        fig, best_xsp, best_dist, best_zone = build_figure(
            mode_key, opp, **grid_kwargs
        )
    st.session_state["fig"]       = fig
    st.session_state["best_xsp"]  = best_xsp
    st.session_state["best_dist"] = best_dist
    st.session_state["best_zone"] = best_zone
    st.session_state["last_key"]  = mode_key

col1, col2, col3 = st.columns(3)
col1.metric("Best xSP",  f"{st.session_state['best_xsp']*100:.1f}%")
col2.metric("Distance",  f"{st.session_state['best_dist']:.1f} m")
col3.metric("Best zone", st.session_state["best_zone"])

st.pyplot(st.session_state["fig"], use_container_width=True)

st.markdown("""
<p style='text-align:center; color:#555555; font-size:12px'>
Built with StatsBomb open data · xSP model trained on 9 competitions
</p>
""", unsafe_allow_html=True)
