# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading models and profiles...")
fk_model     = joblib.load(os.path.join(BASE, "xsp_model.pkl"))
corner_model = joblib.load(os.path.join(BASE, "xsp_corner_model.pkl"))
FK_COLS      = fk_model.get_booster().feature_names
CORNER_COLS  = corner_model.get_booster().feature_names
fk_teams_dict       = joblib.load(os.path.join(BASE, "fk_opp_profiles.pkl"))
corner_teams_dict   = joblib.load(os.path.join(BASE, "corner_opp_profiles.pkl"))
fk_teams_sorted     = joblib.load(os.path.join(BASE, "fk_teams_list.pkl"))
corner_teams_sorted = joblib.load(os.path.join(BASE, "corner_teams_list.pkl"))
print("All loaded!")

GOAL_X      = 120.0
GOAL_Y      = 40.0
GOAL_W      = 7.32
FINAL_THIRD = 80.0
BOX_ZONE    = 102.0
MODE        = ["FK"]

def get_zone_label(sbx, sby):
    depth = "Box zone" if sbx >= BOX_ZONE else "Deep zone"
    lane  = (
        "Left wide"  if sby < 16.8 else
        "Left half"  if sby < 29.6 else
        "Central"    if sby < 50.4 else
        "Right half" if sby < 63.2 else
        "Right wide"
    )
    return depth + " - " + lane

def compute_xsp(x, y, mode, params):
    if mode == "FK" and x >= BOX_ZONE:
        return 0.0
    dx      = GOAL_X - x
    dy      = y - GOAL_Y
    angle   = abs(np.degrees(np.arctan2(GOAL_W*dx, dx**2+dy**2-(GOAL_W/2)**2)))
    lateral = abs(y - GOAL_Y)
    zxd = 1 if x < BOX_ZONE else 0
    zxb = 1 if x >= BOX_ZONE else 0
    zlw = 1 if y < 16.8 else 0
    zlh = 1 if 16.8 <= y < 29.6 else 0
    zc  = 1 if 29.6 <= y < 50.4 else 0
    zrh = 1 if 50.4 <= y < 63.2 else 0
    zrw = 1 if y >= 63.2 else 0
    if mode == "FK":
        row = pd.DataFrame([{
            "x": x, "y": y,
            "angle_to_goal": angle, "lateral_offset": lateral,
            "shot_end_y": y, "shot_end_z": params["end_z"],
            "is_foot": 1, "is_head": 0,
            "is_first_time": 0, "is_deflected": 0, "under_pressure": 0,
            "ff_n_defenders": params["defenders"],
            "ff_n_attackers": params["attackers"],
            "ff_gk_present": 1,
            "ff_wall_size": params["wall"],
            "minute": 45, "xg": 0.05,
            "zone_x_Deep zone": zxd,
            "zone_y_Left wide": zlw, "zone_y_Left half": zlh,
            "zone_y_Central": zc, "zone_y_Right half": zrh,
            "zone_y_Right wide": zrw,
        }])
        return float(fk_model.predict_proba(row[FK_COLS])[0, 1])
    else:
        dval   = int(round(params["delivery_val"]))
        is_ins = 1 if dval == 0 else 0
        is_str = 1 if dval == 1 else 0
        is_out = 1 if dval == 2 else 0
        left_c = 1 if round(params["side_val"]) == 0 else 0
        row = pd.DataFrame([{
            "x": x, "y": y,
            "angle_to_goal": angle, "lateral_offset": lateral,
            "shot_end_y": y, "shot_end_z": params["end_z"],
            "is_head": 1, "is_aerial_won": 0, "is_open_goal": 0,
            "is_first_time": 0, "is_deflected": 0, "under_pressure": 0,
            "ff_n_defenders": params["defenders_box"] + 2,
            "ff_n_attackers": params["attackers_box"],
            "ff_defenders_box": params["defenders_box"],
            "is_inswinging": is_ins, "is_outswinging": is_out,
            "is_straight": is_str, "is_left_corner": left_c,
            "minute": 45, "xg": 0.07,
            "zone_x_Box zone": zxb,
            "zone_y_Left wide": zlw, "zone_y_Left half": zlh,
            "zone_y_Central": zc, "zone_y_Right half": zrh,
            "zone_y_Right wide": zrw,
        }])
        return float(corner_model.predict_proba(row[CORNER_COLS])[0, 1])

def generate_grid(mode, params, resolution=50):
    x_vals = np.linspace(80, 118, resolution)
    y_vals = np.linspace(2, 78, resolution)
    grid   = np.zeros((len(y_vals), len(x_vals)))
    for j, x in enumerate(x_vals):
        for i, y in enumerate(y_vals):
            try:
                grid[i, j] = compute_xsp(x, y, mode, params)
            except:
                grid[i, j] = 0
    return x_vals, y_vals, grid

fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor("#1c2b1c")

ax_pitch = fig.add_axes([0.01, 0.08, 0.50, 0.86])
pitch = VerticalPitch(
    pitch_type="statsbomb", pitch_color="#2d6a2d",
    line_color="white", stripe=False,
    goal_type="box", linewidth=1.5, half=True
)
pitch.draw(ax=ax_pitch)
ax_pitch.set_facecolor("#1c2b1c")

DEFAULT_FK = dict(wall=5, attackers=7, defenders=6, end_z=1.2)
x_vals, y_vals, grid = generate_grid("FK", DEFAULT_FK)

cmap = mcolors.LinearSegmentedColormap.from_list(
    "xsp", ["#1a4a1a","#2d8a2d","#f5c518","#e8531c","#c0392b"]
)
norm = mcolors.Normalize(vmin=0, vmax=0.35)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
heatmap = ax_pitch.pcolormesh(
    Y_grid, X_grid, grid,
    cmap=cmap, norm=norm, alpha=0.75, zorder=3, shading="gouraud"
)

fk_box_mask = ax_pitch.add_patch(Rectangle(
    (0, BOX_ZONE), 80, 120 - BOX_ZONE,
    facecolor="#1c2b1c", alpha=0.65, zorder=4, linewidth=0
))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_pitch, shrink=0.35, pad=0.02)
cbar.set_label("xSP score", color="white", fontsize=9)
cbar.ax.tick_params(colors="white")

ax_pitch.axhline(FINAL_THIRD, color="white", linewidth=1.5, linestyle="--", alpha=0.7, zorder=6)
ax_pitch.axhline(BOX_ZONE, color="white", linewidth=0.9, linestyle="--", alpha=0.5, zorder=6)
for xl in [16.8, 29.6, 50.4, 63.2]:
    ax_pitch.plot([xl,xl],[FINAL_THIRD,120], color="white", linewidth=0.7, linestyle="--", alpha=0.4, zorder=6)

lane_centers = [8.4,23.2,40.0,56.8,71.6]
zone_names   = ["Left wide","Left half","Central","Right half","Right wide"]
deep_mid     = (FINAL_THIRD+BOX_ZONE)/2
box_mid      = (BOX_ZONE+120)/2
for xc, name in zip(lane_centers, zone_names):
    ax_pitch.text(xc, deep_mid, name, color="white", fontsize=7, ha="center", va="center", alpha=0.6, zorder=7)
    ax_pitch.text(xc, box_mid,  name, color="white", fontsize=7, ha="center", va="center", alpha=0.6, zorder=7)
ax_pitch.text(-5, deep_mid, "Deep zone", color="white", fontsize=8, ha="center", va="center", alpha=0.8, rotation=90)
ax_pitch.text(-5, box_mid,  "Box zone",  color="white", fontsize=8, ha="center", va="center", alpha=0.8, rotation=90)

bi = np.unravel_index(grid.argmax(), grid.shape)
best_dot, = ax_pitch.plot(y_vals[bi[0]], x_vals[bi[1]], "*", color="#FFD700", markersize=22, markeredgecolor="white", markeredgewidth=1.5, zorder=10)
info_text = ax_pitch.text(40, 82, "Best xSP: --", color="white", fontsize=9, ha="center",
    bbox=dict(facecolor="black", alpha=0.6, edgecolor="#FFD700", linewidth=1, boxstyle="round,pad=0.4"), zorder=11)
click_dot, = ax_pitch.plot([], [], "o", color="#4FC3F7", markersize=12, markeredgecolor="white", markeredgewidth=2, zorder=12)
click_text = ax_pitch.text(0, 0, "", color="white", fontsize=8, ha="center", va="bottom", zorder=13,
    bbox=dict(facecolor="black", alpha=0.6, edgecolor="#4FC3F7", linewidth=1, boxstyle="round,pad=0.3"))
ax_pitch.text(40, FINAL_THIRD - 3,
    "xSP guide:  Low < 0.05   |   Moderate 0.05 - 0.15   |   High > 0.15",
    color="#aaaaaa", fontsize=7, ha="center", va="top", style="italic", zorder=6)
ax_pitch.set_title("xSP Optimiser - Direct Free Kick", color="#F2A623", fontsize=12, fontweight="bold", pad=12)

ax_btn_fk = fig.add_axes([0.53, 0.91, 0.13, 0.06])
btn_fk = Button(ax_btn_fk, "Free Kick", color="#F2A623", hovercolor="#e09000")
btn_fk.label.set_color("black")
btn_fk.label.set_fontsize(11)
btn_fk.label.set_fontweight("bold")
ax_btn_corner = fig.add_axes([0.67, 0.91, 0.13, 0.06])
btn_corner = Button(ax_btn_corner, "Corner", color="#333333", hovercolor="#4FC3F7")
btn_corner.label.set_color("white")
btn_corner.label.set_fontsize(11)
ax_refresh = fig.add_axes([0.83, 0.91, 0.14, 0.06])
btn_refresh_manual = Button(ax_refresh, "Refresh", color="#333333", hovercolor="#555555")
btn_refresh_manual.label.set_color("white")
btn_refresh_manual.label.set_fontsize(10)

fk_axes    = []
fk_sliders = {}
fk_specs   = [
    ("defenders", "Defenders in box", 1,  11,  1,   6),
    ("attackers", "Attackers",        0,   9,  1,   7),
    ("wall",      "Wall defenders",   0,  10,  1,   5),
    ("end_z",     "Shot height (m)",  0.1, 2.8, 0.1, 1.2),
]
fk_title = fig.text(0.53, 0.875, "Free Kick parameters", color="#F2A623", fontsize=10, fontweight="bold")
for i, (key, lbl, vmin, vmax, step, vdef) in enumerate(fk_specs):
    ax_sl = fig.add_axes([0.55, 0.79 - i*0.10, 0.40, 0.025])
    ax_sl.set_facecolor("#2a2a2a")
    sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vdef, valstep=step, color="#F2A623", track_color="#444")
    sl.label.set_color("white")
    sl.label.set_fontsize(9)
    sl.valtext.set_color("#F2A623")
    sl.valtext.set_fontsize(9)
    fk_sliders[key] = sl
    fk_axes.append(ax_sl)

fk_opp_title = fig.text(0.53, 0.355, "Opposition team", color="#aaaaaa", fontsize=8, style="italic")
ax_fk_drop_display = fig.add_axes([0.55, 0.275, 0.40, 0.065])
ax_fk_drop_display.set_facecolor("#1a1a1a")
ax_fk_drop_display.set_xlim(0, 1)
ax_fk_drop_display.set_ylim(0, 1)
ax_fk_drop_display.axis("off")
fk_selected_text = ax_fk_drop_display.text(0.5, 0.5, "Average", color="white", fontsize=10, ha="center", va="center",
    bbox=dict(facecolor="#2a2a2a", edgecolor="#F2A623", linewidth=1.5, boxstyle="round,pad=0.5"))
ax_fk_prev = fig.add_axes([0.55,  0.275, 0.05, 0.065])
ax_fk_next = fig.add_axes([0.905, 0.275, 0.05, 0.065])
btn_fk_prev = Button(ax_fk_prev, "<", color="#2a2a2a", hovercolor="#555")
btn_fk_next = Button(ax_fk_next, ">", color="#2a2a2a", hovercolor="#555")
for b in [btn_fk_prev, btn_fk_next]:
    b.label.set_color("#F2A623")
    b.label.set_fontsize(14)
    b.label.set_fontweight("bold")
fk_list_axes  = []
fk_list_btns  = []
FK_LIST_SIZE  = 5
fk_list_start = [0]
for i in range(FK_LIST_SIZE):
    ax_li = fig.add_axes([0.55, 0.218 - i*0.036, 0.40, 0.030])
    ax_li.set_facecolor("#1a1a1a")
    btn_li = Button(ax_li, "", color="#1a1a1a", hovercolor="#2a3a2a")
    btn_li.label.set_color("white")
    btn_li.label.set_fontsize(8)
    btn_li.label.set_ha("center")
    fk_list_axes.append(ax_li)
    fk_list_btns.append(btn_li)
fk_axes += [ax_fk_drop_display, ax_fk_prev, ax_fk_next] + fk_list_axes
all_fk_extra = [fk_title, fk_opp_title]

corner_axes    = []
corner_sliders = {}
corner_title = fig.text(0.53, 0.875, "Corner parameters", color="#4FC3F7", fontsize=10, fontweight="bold")
corner_title.set_visible(False)
corner_specs = [
    ("defenders_box", "Defenders in box", 0,  11, 1,  6),
    ("attackers_box", "Attackers",        0,   9, 1,  7),
    ("end_z",         "Shot height (m)",  0.1, 2.8, 0.1, 1.2),
]
for i, (key, lbl, vmin, vmax, step, vdef) in enumerate(corner_specs):
    ax_sl = fig.add_axes([0.55, 0.79 - i*0.10, 0.40, 0.025])
    ax_sl.set_facecolor("#1a3a3a")
    sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vdef, valstep=step, color="#4FC3F7", track_color="#444")
    sl.label.set_color("white")
    sl.label.set_fontsize(9)
    sl.valtext.set_color("#4FC3F7")
    sl.valtext.set_fontsize(9)
    corner_sliders[key] = sl
    ax_sl.set_visible(False)
    corner_axes.append(ax_sl)

ax_del = fig.add_axes([0.55, 0.46, 0.40, 0.025])
ax_del.set_facecolor("#1a3a3a")
sl_delivery = Slider(ax_del, "Delivery type", 0, 2, valinit=0, valstep=1, color="#4FC3F7", track_color="#444")
sl_delivery.label.set_color("white")
sl_delivery.label.set_fontsize(9)
sl_delivery.valtext.set_visible(False)
ax_del.set_visible(False)
corner_axes.append(ax_del)
lbl_del_title = fig.text(0.53, 0.500, "Delivery type", color="#aaaaaa", fontsize=8, style="italic")
lbl_ins  = fig.text(0.558, 0.448, "Inswinging",  color="#4FC3F7", fontsize=8, ha="left")
lbl_str  = fig.text(0.735, 0.448, "Straight",    color="#4FC3F7", fontsize=8, ha="center")
lbl_out  = fig.text(0.910, 0.448, "Outswinging", color="#4FC3F7", fontsize=8, ha="right")
for lbl in [lbl_del_title, lbl_ins, lbl_str, lbl_out]:
    lbl.set_visible(False)
    corner_axes.append(lbl)

ax_side_sl = fig.add_axes([0.55, 0.37, 0.40, 0.025])
ax_side_sl.set_facecolor("#1a3a3a")
sl_side = Slider(ax_side_sl, "Corner side", 0, 1, valinit=0, valstep=1, color="#4FC3F7", track_color="#444")
sl_side.label.set_color("white")
sl_side.label.set_fontsize(9)
sl_side.valtext.set_visible(False)
ax_side_sl.set_visible(False)
corner_axes.append(ax_side_sl)
lbl_side_title = fig.text(0.53, 0.408, "Corner side",   color="#aaaaaa", fontsize=8, style="italic")
lbl_left_c  = fig.text(0.558, 0.358, "Left corner",  color="#4FC3F7", fontsize=8, ha="left")
lbl_right_c = fig.text(0.910, 0.358, "Right corner", color="#4FC3F7", fontsize=8, ha="right")
for lbl in [lbl_side_title, lbl_left_c, lbl_right_c]:
    lbl.set_visible(False)
    corner_axes.append(lbl)

c_opp_title = fig.text(0.53, 0.295, "Opposition team", color="#aaaaaa", fontsize=8, style="italic")
c_opp_title.set_visible(False)
ax_c_drop_display = fig.add_axes([0.55, 0.215, 0.40, 0.065])
ax_c_drop_display.set_facecolor("#1a1a1a")
ax_c_drop_display.set_xlim(0, 1)
ax_c_drop_display.set_ylim(0, 1)
ax_c_drop_display.axis("off")
ax_c_drop_display.set_visible(False)
c_selected_text = ax_c_drop_display.text(0.5, 0.5, "Average", color="white", fontsize=10, ha="center", va="center",
    bbox=dict(facecolor="#1a3a3a", edgecolor="#4FC3F7", linewidth=1.5, boxstyle="round,pad=0.5"))
ax_c_prev = fig.add_axes([0.55,  0.215, 0.05, 0.065])
ax_c_next = fig.add_axes([0.905, 0.215, 0.05, 0.065])
btn_c_prev = Button(ax_c_prev, "<", color="#1a3a3a", hovercolor="#444")
btn_c_next = Button(ax_c_next, ">", color="#1a3a3a", hovercolor="#444")
for b in [btn_c_prev, btn_c_next]:
    b.label.set_color("#4FC3F7")
    b.label.set_fontsize(14)
    b.label.set_fontweight("bold")
ax_c_prev.set_visible(False)
ax_c_next.set_visible(False)
c_list_axes  = []
c_list_btns  = []
c_list_start = [0]
for i in range(FK_LIST_SIZE):
    ax_li = fig.add_axes([0.55, 0.158 - i*0.036, 0.40, 0.030])
    ax_li.set_facecolor("#1a1a1a")
    btn_li = Button(ax_li, "", color="#1a1a1a", hovercolor="#1a3a3a")
    btn_li.label.set_color("white")
    btn_li.label.set_fontsize(8)
    c_list_axes.append(ax_li)
    c_list_btns.append(btn_li)
    ax_li.set_visible(False)
corner_axes += [ax_c_drop_display, ax_c_prev, ax_c_next] + c_list_axes
all_corner_extra = [corner_title, lbl_del_title, lbl_ins, lbl_str, lbl_out, lbl_side_title, lbl_left_c, lbl_right_c, c_opp_title]

def update_fk_list():
    start = fk_list_start[0]
    for i, (ax_li, btn_li) in enumerate(zip(fk_list_axes, fk_list_btns)):
        idx = start + i
        if idx < len(fk_teams_sorted):
            name    = fk_teams_sorted[idx]
            current = fk_selected_text.get_text()
            color   = "#2a4a2a" if name == current else "#1a1a1a"
            ax_li.set_facecolor(color)
            btn_li.color = color
            btn_li.label.set_text(name)
            btn_li.label.set_color("#F2A623" if name == current else "white")
        else:
            btn_li.label.set_text("")
        ax_li.set_visible(True)
    fig.canvas.draw_idle()

def update_c_list():
    start = c_list_start[0]
    for i, (ax_li, btn_li) in enumerate(zip(c_list_axes, c_list_btns)):
        idx = start + i
        if idx < len(corner_teams_sorted):
            name    = corner_teams_sorted[idx]
            current = c_selected_text.get_text()
            color   = "#1a3a4a" if name == current else "#1a1a1a"
            ax_li.set_facecolor(color)
            btn_li.color = color
            btn_li.label.set_text(name)
            btn_li.label.set_color("#4FC3F7" if name == current else "white")
        else:
            btn_li.label.set_text("")
        ax_li.set_visible(True)
    fig.canvas.draw_idle()

def apply_fk_opp(name):
    fk_selected_text.set_text(name)
    if name == "Average":
        fk_sliders["wall"].set_val(5)
        fk_sliders["defenders"].set_val(6)
        fk_sliders["attackers"].set_val(7)
    elif name in fk_teams_dict:
        p = fk_teams_dict[name]
        fk_sliders["wall"].set_val(p["wall"])
        fk_sliders["defenders"].set_val(p["defenders"])
        fk_sliders["attackers"].set_val(p["attackers"])
    update_fk_list()
    refresh()

def apply_c_opp(name):
    c_selected_text.set_text(name)
    if name == "Average":
        corner_sliders["defenders_box"].set_val(6)
        corner_sliders["attackers_box"].set_val(7)
    elif name in corner_teams_dict:
        p = corner_teams_dict[name]
        corner_sliders["defenders_box"].set_val(p["defenders_box"])
        corner_sliders["attackers_box"].set_val(p["attackers_box"])
    update_c_list()
    refresh()

def make_fk_click(i):
    def handler(event):
        idx = fk_list_start[0] + i
        if idx < len(fk_teams_sorted):
            apply_fk_opp(fk_teams_sorted[idx])
    return handler

def make_c_click(i):
    def handler(event):
        idx = c_list_start[0] + i
        if idx < len(corner_teams_sorted):
            apply_c_opp(corner_teams_sorted[idx])
    return handler

for i, btn_li in enumerate(fk_list_btns):
    btn_li.on_clicked(make_fk_click(i))
for i, btn_li in enumerate(c_list_btns):
    btn_li.on_clicked(make_c_click(i))

def fk_prev(event):
    fk_list_start[0] = max(0, fk_list_start[0] - FK_LIST_SIZE)
    update_fk_list()
def fk_next(event):
    fk_list_start[0] = min(len(fk_teams_sorted) - FK_LIST_SIZE, fk_list_start[0] + FK_LIST_SIZE)
    update_fk_list()
def c_prev(event):
    c_list_start[0] = max(0, c_list_start[0] - FK_LIST_SIZE)
    update_c_list()
def c_next(event):
    c_list_start[0] = min(len(corner_teams_sorted) - FK_LIST_SIZE, c_list_start[0] + FK_LIST_SIZE)
    update_c_list()

btn_fk_prev.on_clicked(fk_prev)
btn_fk_next.on_clicked(fk_next)
btn_c_prev.on_clicked(c_prev)
btn_c_next.on_clicked(c_next)

def show_fk():
    MODE[0] = "FK"
    btn_fk.color     = "#F2A623"
    btn_corner.color = "#333333"
    btn_fk.label.set_color("black")
    btn_corner.label.set_color("white")
    fk_box_mask.set_visible(True)
    for a in fk_axes:
        try: a.set_visible(True)
        except: pass
    for lbl in all_fk_extra:
        lbl.set_visible(True)
    for a in corner_axes:
        try: a.set_visible(False)
        except: pass
    for lbl in all_corner_extra:
        lbl.set_visible(False)
    click_dot.set_data([], [])
    click_text.set_text("")
    update_fk_list()

def show_corner():
    MODE[0] = "Corner"
    btn_fk.color     = "#333333"
    btn_corner.color = "#4FC3F7"
    btn_fk.label.set_color("white")
    btn_corner.label.set_color("black")
    fk_box_mask.set_visible(False)
    for a in fk_axes:
        try: a.set_visible(False)
        except: pass
    for lbl in all_fk_extra:
        lbl.set_visible(False)
    for a in corner_axes:
        try: a.set_visible(True)
        except: pass
    for lbl in all_corner_extra:
        lbl.set_visible(True)
    click_dot.set_data([], [])
    click_text.set_text("")
    update_c_list()

def get_params():
    mode = MODE[0]
    if mode == "FK":
        return mode, dict(
            defenders = int(fk_sliders["defenders"].val),
            attackers = int(fk_sliders["attackers"].val),
            wall      = int(fk_sliders["wall"].val),
            end_z     = float(fk_sliders["end_z"].val),
        )
    else:
        return mode, dict(
            defenders_box = int(corner_sliders["defenders_box"].val),
            attackers_box = int(corner_sliders["attackers_box"].val),
            end_z         = float(corner_sliders["end_z"].val),
            delivery_val  = float(sl_delivery.val),
            side_val      = float(sl_side.val),
        )

def refresh(event=None):
    info_text.set_text("Computing...")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    mode, params = get_params()
    _, _, new_grid = generate_grid(mode, params, resolution=50)
    heatmap.set_array(new_grid.ravel())
    bi    = np.unravel_index(new_grid.argmax(), new_grid.shape)
    bx    = x_vals[bi[1]]
    by    = y_vals[bi[0]]
    bdist = np.sqrt((GOAL_X-bx)**2 + (by-GOAL_Y)**2)
    bzone = get_zone_label(bx, by)
    best_dot.set_data([by], [bx])
    info_text.set_text("Best xSP: " + str(round(new_grid.max(), 4)) + " | Dist: " + str(round(bdist, 1)) + "m | " + bzone)
    dnames = {0:"Inswinging", 1:"Straight", 2:"Outswinging"}
    snames = {0:"Left", 1:"Right"}
    if mode == "FK":
        opp      = fk_selected_text.get_text()
        subtitle = " | vs " + opp + " | Wall:" + str(params["wall"]) + " | H:" + str(params["end_z"]) + "m"
        color    = "#F2A623"
        label    = "Direct Free Kick"
    else:
        opp  = c_selected_text.get_text()
        dv   = int(round(params["delivery_val"]))
        sv   = int(round(params["side_val"]))
        subtitle = " | vs " + opp + " | " + dnames.get(dv,"?") + " | " + snames.get(sv,"?") + " corner"
        color    = "#4FC3F7"
        label    = "Corner"
    ax_pitch.set_title("xSP Optimiser - " + label + subtitle, color=color, fontsize=10, fontweight="bold", pad=12)
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes != ax_pitch:
        return
    sbx = event.ydata
    sby = event.xdata
    if sbx is None or sby is None:
        return
    if sbx < FINAL_THIRD or sbx > 120 or sby < 0 or sby > 80:
        return
    mode, params = get_params()
    if mode == "FK" and sbx >= BOX_ZONE:
        click_dot.set_data([], [])
        click_text.set_text("")
        fig.canvas.draw_idle()
        return
    xsp   = compute_xsp(sbx, sby, mode, params)
    dist  = np.sqrt((GOAL_X-sbx)**2 + (sby-GOAL_Y)**2)
    zone  = get_zone_label(sbx, sby)
    qual  = "High" if xsp > 0.15 else "Moderate" if xsp > 0.05 else "Low"
    click_dot.set_data([sby], [sbx])
    click_text.set_position((sby, sbx+2))
    click_text.set_text("xSP: " + str(round(xsp,3)) + " [" + qual + "] | " + str(round(dist,1)) + "m | " + zone)
    fig.canvas.draw_idle()


btn_fk.on_clicked(lambda e: [show_fk(), refresh()])
btn_corner.on_clicked(lambda e: [show_corner(), refresh()])
btn_refresh_manual.on_clicked(refresh)
fig.canvas.mpl_connect("button_press_event", on_click)

fig.text(0.26, 0.97, "xSP Optimiser - Free Kick & Corner", color="white", fontsize=14, fontweight="bold", ha="center")
fig.text(0.26, 0.94, "Select mode | Pick opposition | Adjust sliders | Click pitch", color="#aaaaaa", fontsize=8, ha="center")

ax_loading = fig.add_axes([0.15, 0.35, 0.70, 0.30])
ax_loading.set_facecolor("#1c2b1c")
ax_loading.axis("off")
ax_loading.text(0.5, 0.65, "⚽  xSP Optimiser", color="#F2A623", fontsize=28, fontweight="bold", ha="center", va="center", transform=ax_loading.transAxes)
ax_loading.text(0.5, 0.35, "Loading models and data...", color="#aaaaaa", fontsize=13, ha="center", va="center", transform=ax_loading.transAxes)
plt.pause(0.05)
ax_loading.set_visible(False)
show_fk()
refresh()
plt.show()
print("Closed.")