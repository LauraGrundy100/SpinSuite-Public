"""Static field-direction scan: the recombination yield against the direction of the field in the molecule's own
frame, with no drive, for Daniel's classical and CISS flavin-tryptophan pairs.

For each pair: the yield on a latitude-longitude grid of field directions (mapped by figures.py), the lowest and
highest yield and their directions (best grid point, then polished by Nelder-Mead),
the contrast (highest minus lowest), and the angle from each of those directions to the nearest FN5 principal axis
(Daniel: the extremes should lie close to FN5 eigenvectors). Also the yields with the field along each FN5 axis,
his shortcut for later runs. One process, about 10 minutes; runs as a script or cell by cell.
Each grid point's yields are stored in scan order (theta slowest): reshape to (theta, phi) with the map angles.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize

# Make model.py, engine.py and results.py importable, whether run as a script or cell by cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from engine import field_direction_yields, recombination_yield
from model import flavin_tryptophan_hypothesis
from results import format_duration, new_run_id, save_quick_look, save_run

# %% Settings

FIELD_STRENGTH_uT = 50.0  # geomagnetic field strength (µT). Assumption: Daniel gave no strength
# Daniel's two pairs (assumptions to confirm with him are listed in flavin_tryptophan_hypothesis, model.py):
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)  # classical: chi 0, k_r = k_f = 1 per µs
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)  # CISS: chi pi/2 (maximal),
#                                                                                       k_r = 10, k_f = 1 per µs

TIMES_US = np.linspace(0, 12, 4001)  # yield window (µs), 3 ns grid. About 12 lifetimes of 1/k_f (Daniel: 10 to 12).
#                                      Scratch 2026-10-15: a 0.5 ns grid changes the yields by about 1e-6
MAX_STEP_US = 0.003  # with no drive every step is exact (no drive kicks to split), so this only sets the cost

SCAN_STEP_DEG = 5.0  # field directions: a latitude-longitude grid in the molecule's frame, theta 0 to 180 and phi
#                     -180 to 180 (not repeated) in steps of this many degrees (5: 37 x 72 = 2664 directions, about
#                     10 min for both pairs). The grid is what figures.py maps. Version 1 used 590 Lebedev directions
REFINE_EXTREMES = True  # polish the lowest and highest directions by Nelder-Mead on (theta, phi) from the best grid point
REFINE_TOLERANCE_RAD = 1e-3  # stop polishing when the angles move less than this (about 0.06 degrees)

########################################################################################################
########################################################################################################

# %% Yields for a list of directions, polishing an extreme, angles to the FN5 axes

def unit_vector(polar_angle_rad, azimuth_rad):
    """Direction (θ, φ) in rad as a unit vector (x, y, z) in the molecule's frame."""
    return np.array([np.sin(polar_angle_rad) * np.cos(azimuth_rad), np.sin(polar_angle_rad) * np.sin(azimuth_rad),
                     np.cos(polar_angle_rad)])


def static_recombination_yields(hypothesis, field_directions):
    """Recombination yield and unreacted fraction, one each per field direction ((θ array, φ array) in rad), no drive."""
    singlet_yields, triplet_yields = field_direction_yields(hypothesis, None, TIMES_US, MAX_STEP_US, field_directions)
    return recombination_yield(hypothesis, singlet_yields), 1.0 - singlet_yields - triplet_yields


def polished_extreme(hypothesis, polar_angle_rad, azimuth_rad, sign):
    """Lowest (sign +1) or highest (sign −1) recombination yield near a starting direction (rad), by Nelder-Mead on
    (θ, φ). Returns θ, φ (rad) and the yield there."""
    def signed_yield(angles_rad):
        yield_here, _ = static_recombination_yields(hypothesis, ([angles_rad[0]], [angles_rad[1]]))
        return sign * yield_here[0]

    result = scipy.optimize.minimize(signed_yield, [polar_angle_rad, azimuth_rad], method="Nelder-Mead",
                                     options={"xatol": REFINE_TOLERANCE_RAD, "fatol": 1e-9})
    return result.x[0], result.x[1], sign * result.fun


def angles_to_axes_deg(direction, axes):
    """Angle in degrees (0 to 90) between a direction and each axis (columns of axes); an axis has no sign."""
    return np.degrees(np.arccos(np.clip(np.abs(axes.T @ direction), 0.0, 1.0)))


def scan_one_pair(hypothesis, scan_directions, fn5_axes):
    """Everything reported for one pair, as a dict: yields on the grid, the lowest and highest (polished), the
    contrast, their angles to the FN5 axes, and the yields with the field along each FN5 axis (both ways)."""
    grid_yields, unreacted = static_recombination_yields(hypothesis, scan_directions)
    outcome = {"recombination_yields": grid_yields, "largest_unreacted": np.max(np.abs(unreacted))}
    for label, index, sign in (("lowest", np.argmin(grid_yields), 1), ("highest", np.argmax(grid_yields), -1)):
        polar_angle_rad, azimuth_rad = scan_directions[0][index], scan_directions[1][index]
        yield_there = grid_yields[index]
        if REFINE_EXTREMES:
            polar_angle_rad, azimuth_rad, yield_there = polished_extreme(hypothesis, polar_angle_rad, azimuth_rad, sign)
        direction = unit_vector(polar_angle_rad, azimuth_rad)
        outcome.update({f"{label}_yield": yield_there, f"{label}_direction": direction,
                        f"{label}_polar_angle_rad": polar_angle_rad, f"{label}_azimuth_rad": azimuth_rad,
                        f"{label}_angles_to_fn5_axes_deg": angles_to_axes_deg(direction, fn5_axes)})
    outcome["contrast"] = outcome["highest_yield"] - outcome["lowest_yield"]

    axis_directions = np.concatenate([fn5_axes, -fn5_axes], axis=1)  # each FN5 axis both ways: 6 directions
    axis_polar_angles_rad = np.arccos(np.clip(axis_directions[2], -1.0, 1.0))
    axis_azimuths_rad = np.arctan2(axis_directions[1], axis_directions[0])
    outcome["fn5_axis_yields"], _ = static_recombination_yields(hypothesis, (axis_polar_angles_rad, axis_azimuths_rad))
    return outcome


def print_summary(label, hypothesis, outcome):
    """Plain-text report for one pair."""
    print(f"Pair {label}: {hypothesis.name}")
    for extreme in ("lowest", "highest"):
        nearest = np.argmin(outcome[f"{extreme}_angles_to_fn5_axes_deg"])
        print(f"  {extreme:7s} yield {outcome[f'{extreme}_yield']:.6f} with the field along "
              f"{np.round(outcome[f'{extreme}_direction'], 3).tolist()}: "
              f"{outcome[f'{extreme}_angles_to_fn5_axes_deg'][nearest]:.1f} deg from FN5 axis {nearest + 1} "
              f"(angles to axes 1-3: {np.round(outcome[f'{extreme}_angles_to_fn5_axes_deg'], 1).tolist()} deg)")
    fn5_axis_yields = outcome["fn5_axis_yields"]
    print(f"  contrast (highest - lowest) {outcome['contrast']:.6f}; over the six FN5 axis directions only "
          f"{np.max(fn5_axis_yields) - np.min(fn5_axis_yields):.6f}")
    print(f"  yields along FN5 axes 1-3 (+ then -): {np.round(fn5_axis_yields, 6).tolist()}")
    print(f"  largest unreacted fraction at {TIMES_US[-1]:g} us: {outcome['largest_unreacted']:.1e}")


# %% Run both pairs, print, save

if __name__ == "__main__":
    print(f"Zeeman frequency used: {HYPOTHESIS_A.zeeman_frequency_MHz:.6f} MHz at {FIELD_STRENGTH_uT} uT "
          f"(SciPy CODATA free-electron gyromagnetic ratio)")
    fn5_eigenvalues_MHz, fn5_axes = np.linalg.eigh(HYPOTHESIS_A.nuclei[0].hyperfine_tensor_MHz)  # columns = axes
    print(f"FN5 principal values (MHz), axes 1-3: {np.round(fn5_eigenvalues_MHz, 3).tolist()}")
    map_polar_angles_rad = np.radians(np.arange(0.0, 180.0 + 1e-9, SCAN_STEP_DEG))
    map_azimuths_rad = np.radians(np.arange(-180.0, 180.0 - 1e-9, SCAN_STEP_DEG))
    grid_polar, grid_azimuth = np.meshgrid(map_polar_angles_rad, map_azimuths_rad, indexing="ij")  # (theta, phi)
    scan_polar_angles_rad, scan_azimuths_rad = grid_polar.ravel(), grid_azimuth.ravel()
    print(f"{len(scan_polar_angles_rad)} field directions ({len(map_polar_angles_rad)} theta x "
          f"{len(map_azimuths_rad)} phi)")

    outcome = {"fn5_eigenvalues_MHz": fn5_eigenvalues_MHz, "fn5_axes": fn5_axes,
               "scan_polar_angles_rad": scan_polar_angles_rad, "scan_azimuths_rad": scan_azimuths_rad,
               "map_polar_angles_rad": map_polar_angles_rad, "map_azimuths_rad": map_azimuths_rad,
               "zeeman_frequency_used_MHz": HYPOTHESIS_A.zeeman_frequency_MHz}
    for label, hypothesis in (("A", HYPOTHESIS_A), ("B", HYPOTHESIS_B)):
        start = time.perf_counter()
        pair_outcome = scan_one_pair(hypothesis, (scan_polar_angles_rad, scan_azimuths_rad), fn5_axes)
        print_summary(label, hypothesis, pair_outcome)
        print(f"  ({format_duration(time.perf_counter() - start)})", flush=True)
        outcome.update({f"{name}_{label}": value for name, value in pair_outcome.items()})

    FORMAT_VERSION = 2  # 1: 590 Lebedev directions (scan_lebedev_order 41); 2: the latitude-longitude grid
    settings = dict(hypothesis_A=repr(HYPOTHESIS_A), hypothesis_B=repr(HYPOTHESIS_B),
                    field_strength_uT=FIELD_STRENGTH_uT, times_us=TIMES_US, max_step_us=MAX_STEP_US,
                    scan_step_deg=SCAN_STEP_DEG, refine_extremes=REFINE_EXTREMES,
                    refine_tolerance_rad=REFINE_TOLERANCE_RAD)
    results_folder = REPO_FOLDER / "results"
    run_id = new_run_id(results_folder, "field_scan")
    save_run(results_folder, run_id, FORMAT_VERSION, settings, outcome)
    save_quick_look(results_folder, run_id, "sorted_yields", "field direction (sorted by yield)", "recombination yield",
                    [(np.arange(len(scan_polar_angles_rad)), np.sort(outcome[f"recombination_yields_{label}"]),
                      f"pair {label}") for label in ("A", "B")])
    print(f"Saved results/{run_id}.npz and results/{run_id}_sorted_yields.png")
