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

st.set_page_config(
    page_title="xSP Optimiser",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Slate + Electric Blue theme ─────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

    :root {
        --bg-primary:   #0d1117;
        --bg-secondary: #161b22;
        --bg-card:      #1c2333;
        --border:       #30363d;
        --accent-blue:  #00b4d8;
        --accent-fk:    #f7931e;
        --accent-c:     #00b4d8;
        --text-primary: #e6edf3;
        --text-muted:   #8b949e;
    }

    /* Global */
    .stApp { background-color: var(--bg-primary) !important; font-family: 'DM Mono', monospace; }
    .stApp * { color: var(--text-primary); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* Main header */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(90deg, #00b4d8, #0077b6) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    [data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--accent-blue) !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Sliders */
    .stSlider label { color: var(--text-muted) !important; font-size: 0.75rem !important; letter-spacing: 0.5px !important; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--accent-blue) !important; }

    /* Radio */
    .stRadio label { color: var(--text-primary) !important; font-size: 0.8rem !important; }
    .stRadio [data-baseweb="radio"] div:first-child { border-color: var(--accent-blue) !important; }

    /* Selectbox */
    .stSelectbox label { color: var(--text-muted) !important; font-size: 0.7rem !important; letter-spacing: 0.5px !important; text-transform: uppercase !important; }
    .stSelectbox [data-baseweb="select"] { background: var(--bg-card) !important; border-color: var(--border) !important; }

    /* Select slider */
    .stSelectSlider label { color: var(--text-muted) !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }

    /* Sidebar title */
    section[data-testid="stSidebar"] h2 {
        font-family: 'Syne', sans-serif !important;
        color: var(--accent-blue) !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        -webkit-text-fill-color: var(--accent-blue) !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* Warning */
    [data-testid="stAlert"] { background: rgba(247,147,30,0.1) !important; border-color: var(--accent-fk) !important; }

    /* Spinner */
    .stSpinner { color: var(--accent-blue) !important; }

    /* Markdown text */
    p { color: var(--text-muted) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
GOAL_X      = 120.0
GOAL_Y      = 40.0
GOAL_W      = 7.32
FINAL_THIRD = 80.0
BOX_ZONE    = 102.0
RESOLUTION  = 35

# Matplotlib theme to match slate UI
PLT_BG      = "#0d1117"
PLT_PITCH   = "#161b22"
PLT_LINE    = "#30363d"
PLT_ACCENT  = "#00b4d8"
PLT_FK      = "#f7931e"
PLT_TEXT    = "#e6edf3"
PLT_MUTED   = "#8b949e"

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

with st.spinner("Loading models..."):
    (fk_model, corner_model, FK_COLS, CORNER_COLS,
     fk_teams_dict, corner_teams_dict,
     fk_teams_sorted, corner_teams_sorted) = load_models()

@st.cache_data
def generate_grid(mode, wall=0, defenders=5, attackers=7, end_z=1.2,
                  defenders_box=6, attackers_box=7,
                  delivery_val=0, side_val=0, resolution=RESOLUTION):
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
            "ff_n_defenders": defenders, "ff_n_attackers": attackers,
            "ff_gk_present": 1, "ff_wall_size": wall,
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
            "ff_n_defenders": defenders_box + 2, "ff_n_attackers": attackers_box,
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

def draw_player_dots(ax, mode, params, bx, by):
    if mode == "FK":
        wall  = params["wall"]
        defs  = params["defenders"]
        atts  = params["attackers"]

        dx   = GOAL_X - bx
        dy   = GOAL_Y - by
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            px, py = -uy, ux
            wall_x = bx + ux * 9.15
            wall_y = by + uy * 9.15
            offsets = np.linspace(-(wall-1)/2, (wall-1)/2, wall) if wall > 0 else []
            for off in offsets:
                ax.plot(wall_y + py * off * 0.8, wall_x + px * off * 0.8,
                        'o', color='#ef4444', markersize=7,
                        markeredgecolor='white', markeredgewidth=0.8, zorder=15)

        def_x = np.linspace(103, 116, max(defs, 1))
        def_y = np.linspace(28, 52, max(defs, 1))
        for i in range(defs):
            ax.plot(def_y[i%len(def_y)], def_x[i%len(def_x)],
                    's', color='#991b1b', markersize=6,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=15)

        att_x = np.linspace(104, 115, max(atts, 1))
        att_y = np.linspace(32, 48, max(atts, 1))
        for i in range(atts):
            ax.plot(att_y[i%len(att_y)], att_x[i%len(att_x)],
                    '^', color='#1d4ed8', markersize=7,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=15)

        for lx, ly, sym, col, lbl in [
            (8,  FINAL_THIRD-4, '●', '#ef4444', f'Wall ({wall})'),
            (25, FINAL_THIRD-4, '■', '#991b1b', f'Def ({defs})'),
            (48, FINAL_THIRD-4, '▲', '#1d4ed8', f'Att ({atts})'),
        ]:
            ax.text(lx, ly, f'{sym} {lbl}', color=col,
                    fontsize=6, fontweight='bold', ha='center', va='center', zorder=14)
    else:
        defs_box = params["defenders_box"]
        atts_box = params["attackers_box"]
        def_x = np.linspace(104, 118, max(defs_box, 1))
        def_y = np.linspace(25, 55, max(defs_box, 1))
        for i in range(defs_box):
            ax.plot(def_y[i%len(def_y)], def_x[i%len(def_x)],
                    's', color='#ef4444', markersize=6,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=15)
        att_x = np.linspace(105, 117, max(atts_box, 1))
        att_y = np.linspace(30, 50, max(atts_box, 1))
        for i in range(atts_box):
            ax.plot(att_y[i%len(att_y)], att_x[i%len(att_x)],
                    '^', color='#1d4ed8', markersize=7,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=15)
        for lx, ly, sym, col, lbl in [
            (18, FINAL_THIRD-4, '■', '#ef4444', f'Def ({defs_box})'),
            (45, FINAL_THIRD-4, '▲', '#1d4ed8', f'Att ({atts_box})'),
        ]:
            ax.text(lx, ly, f'{sym} {lbl}', color=col,
                    fontsize=6, fontweight='bold', ha='center', va='center', zorder=14)

@st.cache_data
def build_figure(mode, opp_name, wall=0, defenders=5, attackers=7, end_z=1.2,
                 defenders_box=6, attackers_box=7, delivery_val=0, side_val=0):

    x_vals, y_vals, grid = generate_grid(
        mode, wall=wall, defenders=defenders, attackers=attackers, end_z=end_z,
        defenders_box=defenders_box, attackers_box=attackers_box,
        delivery_val=delivery_val, side_val=side_val
    )

    # Medium size — 20% smaller than before
    fig = plt.figure(figsize=(9, 7.5))
    fig.patch.set_facecolor(PLT_BG)
    ax = fig.add_axes([0.04, 0.04, 0.84, 0.90])

    pitch = VerticalPitch(
        pitch_type="statsbomb", pitch_color=PLT_PITCH,
        line_color=PLT_LINE, stripe=False,
        goal_type="box", linewidth=1.2, half=True
    )
    pitch.draw(ax=ax)
    ax.set_facecolor(PLT_BG)

    # Electric blue → amber heatmap for FK, blue spectrum for corners
    if mode == "FK":
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "xsp_fk", ["#0d1117", "#1e3a5f", "#0077b6", "#00b4d8", "#f7931e", "#ef4444"]
        )
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "xsp_c", ["#0d1117", "#0c2340", "#023e8a", "#0077b6", "#00b4d8", "#90e0ef"]
        )

    norm = mcolors.Normalize(vmin=0, vmax=0.35)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    ax.pcolormesh(Y_grid, X_grid, grid, cmap=cmap, norm=norm,
                  alpha=0.85, zorder=3, shading="gouraud")

    if mode == "FK":
        ax.add_patch(Rectangle(
            (0, BOX_ZONE), 80, 120 - BOX_ZONE,
            facecolor=PLT_BG, alpha=0.75, zorder=4, linewidth=0
        ))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.35, pad=0.02)
    cbar.set_label("xSP", color=PLT_MUTED, fontsize=8)
    cbar.ax.tick_params(colors=PLT_MUTED, labelsize=7)
    cbar.outline.set_edgecolor(PLT_LINE)

    # Zone lines
    ax.axhline(FINAL_THIRD, color=PLT_LINE, linewidth=1.2, linestyle="--", alpha=0.8, zorder=6)
    ax.axhline(BOX_ZONE, color=PLT_LINE, linewidth=0.8, linestyle="--", alpha=0.6, zorder=6)
    for xl in [16.8, 29.6, 50.4, 63.2]:
        ax.plot([xl,xl],[FINAL_THIRD,120], color=PLT_LINE,
                linewidth=0.6, linestyle="--", alpha=0.5, zorder=6)
    for xl in [36.3, 43.7]:
        ax.plot([xl,xl],[BOX_ZONE,120], color=PLT_ACCENT,
                linewidth=0.7, linestyle=":", alpha=0.6, zorder=6)

    lane_centers = [8.4, 23.2, 33.0, 40.0, 47.0, 56.8, 71.6]
    zone_names   = ["L.wide","L.half","L.central","Central","R.central","R.half","R.wide"]
    deep_mid = (FINAL_THIRD + BOX_ZONE) / 2
    box_mid  = (BOX_ZONE + 120) / 2
    for xc, name in zip(lane_centers, zone_names):
        ax.text(xc, deep_mid, name, color=PLT_MUTED, fontsize=4.5,
                ha="center", va="center", alpha=0.7, zorder=7)
        ax.text(xc, box_mid, name, color=PLT_MUTED, fontsize=4.5,
                ha="center", va="center", alpha=0.7, zorder=7)
    ax.text(-4, deep_mid, "DEEP", color=PLT_MUTED, fontsize=5.5,
            ha="center", va="center", alpha=0.7, rotation=90)
    ax.text(-4, box_mid,  "BOX",  color=PLT_MUTED, fontsize=5.5,
            ha="center", va="center", alpha=0.7, rotation=90)

    # Best spot
    bi    = np.unravel_index(grid.argmax(), grid.shape)
    bx    = x_vals[bi[1]]
    by    = y_vals[bi[0]]
    bdist = np.sqrt((GOAL_X-bx)**2 + (by-GOAL_Y)**2)
    bzone = get_zone_label(bx, by)

    ax.plot(y_vals[bi[0]], x_vals[bi[1]], "*",
            color="#ffd60a", markersize=18,
            markeredgecolor=PLT_BG, markeredgewidth=1.2, zorder=10)
    ax.text(40, 82.5,
            f"★  {grid.max()*100:.1f}%   ·   {bdist:.1f}m   ·   {bzone}",
            color=PLT_TEXT, fontsize=7.5, ha="center",
            bbox=dict(facecolor=PLT_BG, alpha=0.85, edgecolor="#ffd60a",
                      linewidth=1, boxstyle="round,pad=0.4"), zorder=11)
    ax.text(40, FINAL_THIRD - 3,
            "LOW  < 5%   ·   MODERATE  5–15%   ·   HIGH  > 15%",
            color=PLT_MUTED, fontsize=5.5, ha="center", va="top",
            style="italic", zorder=6)

    # Player dots
    params_dots = dict(
        wall=wall, defenders=defenders, attackers=attackers,
        defenders_box=defenders_box, attackers_box=attackers_box
    )
    draw_player_dots(ax, mode, params_dots, bx, by)

    accent = PLT_FK if mode == "FK" else PLT_ACCENT
    label  = "Direct Free Kick" if mode == "FK" else "Corner"
    ax.set_title(f"{label.upper()}  ·  vs {opp_name}",
                 color=accent, fontsize=9, fontweight="bold",
                 pad=8, fontfamily="monospace")

    return fig, float(grid.max()), float(bdist), bzone

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## xSP Optimiser")
    st.markdown("---")
    mode = st.radio("Mode", ["Free Kick", "Corner"], horizontal=True)
    st.markdown("---")

    if mode == "Free Kick":
        st.markdown("### 🟠 FREE KICK")
        opp = st.selectbox("Opposition", fk_teams_sorted, index=0)
        if opp == "Average":
            def_wall, def_def, def_att = 5, 5, 7
        elif opp in fk_teams_dict:
            p = fk_teams_dict[opp]
            def_wall = p["wall"]
            def_def  = min(p["defenders"], 10 - def_wall)
            def_att  = p["attackers"]
        else:
            def_wall, def_def, def_att = 5, 5, 7

        wall      = st.slider("Wall defenders",   0, 10, def_wall)
        defenders = st.slider("Defenders in box", 0, max(1, 10-wall), min(def_def, 10-wall))
        attackers = st.slider("Attackers",        0,  9, def_att)
        end_z     = st.slider("Shot height (m)",  0.1, 2.8, 1.2, step=0.1)

        if wall + defenders > 10:
            st.warning(f"⚠️ Wall + Def = {wall+defenders} > 10 — capped")
            defenders = 10 - wall

        grid_kwargs = dict(wall=wall, defenders=defenders,
                           attackers=attackers, end_z=end_z)
    else:
        st.markdown("### 🔵 CORNER")
        opp = st.selectbox("Opposition", corner_teams_sorted, index=0)
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
        delivery      = st.select_slider("Delivery", ["Inswinging","Straight","Outswinging"])
        side          = st.radio("Corner side", ["Left","Right"], horizontal=True)

        delivery_val = ["Inswinging","Straight","Outswinging"].index(delivery)
        side_val     = 0 if side == "Left" else 1

        grid_kwargs = dict(defenders_box=defenders_box, attackers_box=attackers_box,
                           end_z=end_z, delivery_val=delivery_val, side_val=side_val)

# ── Main ──────────────────────────────────────────────────────────────────
mode_key = "FK" if mode == "Free Kick" else "Corner"

st.markdown("<h1>⚽ xSP Optimiser</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase'>"
    "Expected Set Piece · StatsBomb Open Data · 9 Competitions</p>",
    unsafe_allow_html=True
)
st.markdown("---")

with st.spinner("Computing..."):
    fig, best_xsp, best_dist, best_zone = build_figure(
        mode_key, opp, **grid_kwargs
    )

col1, col2, col3 = st.columns(3)
col1.metric("BEST xSP",  f"{best_xsp*100:.1f}%")
col2.metric("DISTANCE",  f"{best_dist:.1f} m")
col3.metric("BEST ZONE", best_zone)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.pyplot(fig, use_container_width=True)

st.markdown(
    "<p style='text-align:center;font-size:0.65rem;color:#30363d;margin-top:8px'>"
    "● Wall  ■ Defenders  ▲ Attackers  ★ Best xSP position</p>",
    unsafe_allow_html=True
)
