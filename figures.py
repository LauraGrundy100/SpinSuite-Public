from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from results import load_run

# Figures are drawn from the saved runs in results/ and written to figures/ as PNG and PDF.
REPO_FOLDER = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_FOLDER = REPO_FOLDER / "results"
FIGURES_FOLDER = REPO_FOLDER / "figures"


# %% style
# new
PAIR_COLOURS = {"A": "#2a78d6", "B": "#eb6834"}
PAIR_NAMES = {"A": "classical (χ = 0)", "B": "CISS (χ = π/2)"}
INK, SECONDARY_INK, MUTED_INK, GRID_COLOUR = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
YIELD_COLOURMAP = LinearSegmentedColormap.from_list(
    "yield_blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
LINE_WIDTH = 2.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED_INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID_COLOUR, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})


# new
def save_figure(figure, figure_id):
    FIGURES_FOLDER.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(FIGURES_FOLDER / f"{figure_id}.{extension}")
    print(f"Saved figures/{figure_id}.png and .pdf")


# new
def polar_and_azimuth_deg(direction):
    return np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))), np.degrees(np.arctan2(direction[1], direction[0]))


# %% field_map: recombination yield against field direction in the molecule frame, no drive
# new
FIELD_MAP_RUN_ID = "field_scan_2026-10-15_02"

run = load_run(RESULTS_FOLDER / f"{FIELD_MAP_RUN_ID}.npz")
polar_deg, azimuth_deg = np.degrees(run["map_polar_angles_rad"]), np.degrees(run["map_azimuths_rad"])
figure, axes_pair = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
for axes, label in zip(axes_pair, ("A", "B")):
    yield_map = run[f"recombination_yields_{label}"].reshape(len(polar_deg), len(azimuth_deg))
    yield_map = np.concatenate([yield_map, yield_map[:, :1]], axis=1)  # φ = 180° repeats -180°, closes the map
    mesh = axes.pcolormesh(np.append(azimuth_deg, 180.0), polar_deg, yield_map, cmap=YIELD_COLOURMAP, shading="gouraud",
                           rasterized=True)
    colourbar = figure.colorbar(mesh, ax=axes, pad=0.02)
    colourbar.set_label("recombination yield", color=INK)
    colourbar.outline.set_visible(False)
    for axis_number in range(3):
        for sign in (1, -1):
            axis_polar, axis_azimuth = polar_and_azimuth_deg(sign * run["fn5_axes"][:, axis_number])
            axes.plot(axis_azimuth, axis_polar, "o", markersize=9, markerfacecolor="white",
                      markeredgecolor=INK, markeredgewidth=1.0, clip_on=False)
            axes.text(axis_azimuth, axis_polar, str(axis_number + 1), ha="center", va="center", fontsize=6.5,
                      color=INK)
    for extreme, marker in (("lowest", "v"), ("highest", "^")):
        extreme_polar, extreme_azimuth = polar_and_azimuth_deg(run[f"{extreme}_direction_{label}"])
        axes.plot(extreme_azimuth, extreme_polar, marker, markersize=10, markerfacecolor=INK,
                  markeredgecolor="white", markeredgewidth=1.2, clip_on=False)
    axes.set_title(f"Pair {label}: {PAIR_NAMES[label]}, contrast {run[f'contrast_{label}']:.4f}")
    axes.set_xlabel("field azimuth φ (degrees, molecule frame)")
    axes.set_xticks(np.arange(-180, 181, 90))
    axes.set_yticks(np.arange(0, 181, 45))
axes_pair[0].set_ylabel("field polar angle θ (degrees)")
axes_pair[0].invert_yaxis()  # θ = 0 (molecule z) at the top
figure.legend(handles=[
    Line2D([], [], marker="v", linestyle="", markersize=8, markerfacecolor=INK, markeredgecolor="white",
           label="lowest yield"),
    Line2D([], [], marker="^", linestyle="", markersize=8, markerfacecolor=INK, markeredgecolor="white",
           label="highest yield"),
    Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white", markeredgecolor=INK,
           label="FN5 principal axis 1, 2, 3 (both ways)")],
    loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
figure.tight_layout(rect=(0, 0.06, 1, 1))
save_figure(figure, "field_map")


# %% tone_contrast: contrast under one sine at the cap against its frequency
# new
TONE_CONTRAST_RUN_ID = "single_tone_contrast_2026-10-15_01"

run = load_run(RESULTS_FOLDER / f"{TONE_CONTRAST_RUN_ID}.npz")
frequencies_MHz = run["tone_frequencies_MHz"]
figure, (contrast_axes, difference_axes) = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
for label in ("A", "B"):
    contrast_axes.plot(frequencies_MHz, run[f"tone_contrasts_{label}"], color=PAIR_COLOURS[label],
                       linewidth=LINE_WIDTH, label=f"pair {label}, {PAIR_NAMES[label]}")
    contrast_axes.axhline(run[f"static_contrast_{label}"], color=PAIR_COLOURS[label], linewidth=1.0,
                          linestyle="--")
contrast_axes.set_ylabel("contrast C\n(highest − lowest yield)")
contrast_axes.legend(loc="best")
difference = np.abs(run["tone_contrasts_A"] - run["tone_contrasts_B"])
difference_axes.plot(frequencies_MHz, difference, color=INK, linewidth=LINE_WIDTH, label="single tone")
difference_axes.axhline(abs(run["static_contrast_A"] - run["static_contrast_B"]), color=MUTED_INK, linewidth=1.0,
                        linestyle="--", label="no drive")
difference_axes.set_ylabel("|C_A − C_B|")
difference_axes.set_xlabel("drive frequency (MHz)")
difference_axes.legend(loc="best")
for axes in (contrast_axes, difference_axes):
    axes.grid(True, axis="y")
contrast_axes.set_title(f"One sine drive at the cap ({run['amplitude_cap_MHz']:.3f} MHz, placeholder); "
                        "dashed: no drive", loc="left")
figure.tight_layout()
save_figure(figure, "tone_contrast")


# %% overnight_contrast: optimised drive against no drive and the best single tone
# new
OPTIMISED_RUN_ID = "discrimination_2026-10-15_01"

run = load_run(RESULTS_FOLDER / f"{OPTIMISED_RUN_ID}.npz")
tones = load_run(RESULTS_FOLDER / f"{TONE_CONTRAST_RUN_ID}.npz")
best_tone = np.argmax(np.abs(tones["tone_contrasts_A"] - tones["tone_contrasts_B"]))
static_difference = abs(run["static_contrast_A"] - run["static_contrast_B"])
figure, (history_axes, bar_axes) = plt.subplots(1, 2, figsize=(9.0, 3.4))

for random_seed in run["random_seeds"]:  # |C_A - C_B| = -cost
    history = -run[f"random_seed_{random_seed}_cost_history"]
    history_axes.plot(np.arange(len(history)), history, color=SECONDARY_INK, linewidth=1.2, marker="o", markersize=3,
                      alpha=1.0 if random_seed == run["best_random_seed"] else 0.45)
history_axes.axhline(static_difference, color=MUTED_INK, linestyle="--", linewidth=1.0)
history_axes.axhline(abs(tones["tone_contrasts_A"][best_tone] - tones["tone_contrasts_B"][best_tone]), color=INK,
                     linestyle=":", linewidth=1.0)
# check: x = 5 for the labels and "Five searches" below are hard-coded for 5 rounds and 5 seeds
history_axes.text(5, static_difference, "no drive", ha="right", va="bottom", color=MUTED_INK, fontsize=8)
history_axes.text(5, abs(tones["tone_contrasts_A"][best_tone] - tones["tone_contrasts_B"][best_tone]),
                  f"best single tone ({tones['tone_frequencies_MHz'][best_tone]:.1f} MHz)", ha="right", va="bottom",
                  color=INK, fontsize=8)
history_axes.set_xlabel("super-iteration (0 = start)")
history_axes.set_ylabel("|C_A − C_B|")
history_axes.set_title(f"Five searches (best: seed {run['best_random_seed']}, dark)", loc="left")
history_axes.grid(True, axis="y")

labels = ["no drive", f"best single tone\n{tones['tone_frequencies_MHz'][best_tone]:.1f} MHz", "optimised\n(best seed)"]
contrasts_by_case = {"A": [run["static_contrast_A"], tones["tone_contrasts_A"][best_tone], run["driven_contrast_A"]],
                     "B": [run["static_contrast_B"], tones["tone_contrasts_B"][best_tone], run["driven_contrast_B"]]}
positions = np.arange(len(labels))
for offset, label in ((-0.18, "A"), (0.18, "B")):
    bars = bar_axes.bar(positions + offset, contrasts_by_case[label], width=0.34, color=PAIR_COLOURS[label],
                        label=f"pair {label}, {PAIR_NAMES[label]}")
    bar_axes.bar_label(bars, fmt="%.4f", fontsize=7, color=SECONDARY_INK, padding=2)
bar_axes.set_xticks(positions, labels)
bar_axes.set_ylabel("contrast C (highest − lowest yield)")
bar_axes.set_title(f"Cap {run['amplitude_cap_MHz']:.3f} MHz (placeholder)", loc="left")
bar_axes.legend(loc="upper left")
bar_axes.set_ylim(0, 1.2 * max(contrasts_by_case["B"]))
figure.tight_layout()
save_figure(figure, "overnight_contrast")
