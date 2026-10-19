import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from engine import field_direction_yields, recombination_yield
from model import flavin_tryptophan_hypothesis
from results import format_duration, new_run_id, save_quick_look, save_run


# %% settings
# new
FIELD_STRENGTH_uT = 50.0
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)

TIMES_US = np.linspace(0, 12, 4001)
MAX_STEP_US = 0.003  # no drive, so every step is exact and this only sets the cost

SCAN_STEP_DEG = 5.0  # θ 0-180, φ -180 to 180 in the molecule's frame: 37 x 72 directions
REFINE_EXTREMES = True
REFINE_TOLERANCE_RAD = 1e-3


# %% functions
# new
def unit_vector(polar_angle_rad, azimuth_rad):
    return np.array([np.sin(polar_angle_rad) * np.cos(azimuth_rad), np.sin(polar_angle_rad) * np.sin(azimuth_rad),
                     np.cos(polar_angle_rad)])


# new
def static_recombination_yields(hypothesis, field_directions):
    # recombination yield and the fraction still unreacted at the end of the window, per direction
    singlet_yields, triplet_yields = field_direction_yields(hypothesis, None, TIMES_US, MAX_STEP_US, field_directions)
    return recombination_yield(hypothesis, singlet_yields), 1.0 - singlet_yields - triplet_yields


# new
def polished_extreme(hypothesis, polar_angle_rad, azimuth_rad, sign):
    # Nelder-Mead on (θ, φ) from the best grid point; sign +1 for the lowest yield, -1 for the highest
    def signed_yield(angles_rad):
        yield_here, _ = static_recombination_yields(hypothesis, ([angles_rad[0]], [angles_rad[1]]))
        return sign * yield_here[0]

    result = scipy.optimize.minimize(signed_yield, [polar_angle_rad, azimuth_rad], method="Nelder-Mead",
                                     options={"xatol": REFINE_TOLERANCE_RAD, "fatol": 1e-9})
    return result.x[0], result.x[1], sign * result.fun


# new
def angles_to_axes_deg(direction, axes):
    # 0-90 degrees: an axis has no sign
    return np.degrees(np.arccos(np.clip(np.abs(axes.T @ direction), 0.0, 1.0)))


# new
def scan_one_pair(hypothesis, scan_directions, fn5_axes):
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

    # Daniel's shortcut: field along each FN5 principal axis, both ways
    axis_directions = np.concatenate([fn5_axes, -fn5_axes], axis=1)
    axis_polar_angles_rad = np.arccos(np.clip(axis_directions[2], -1.0, 1.0))
    axis_azimuths_rad = np.arctan2(axis_directions[1], axis_directions[0])
    outcome["fn5_axis_yields"], _ = static_recombination_yields(hypothesis, (axis_polar_angles_rad, axis_azimuths_rad))
    return outcome


# new
def print_summary(label, hypothesis, outcome):
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


# %% run
# new
if __name__ == "__main__":
    print(f"Zeeman frequency used: {HYPOTHESIS_A.zeeman_frequency_MHz:.6f} MHz at {FIELD_STRENGTH_uT} uT "
          f"(SciPy CODATA free-electron gyromagnetic ratio)")
    fn5_eigenvalues_MHz, fn5_axes = np.linalg.eigh(HYPOTHESIS_A.nuclei[0].hyperfine_tensor_MHz)
    print(f"FN5 principal values (MHz), axes 1-3: {np.round(fn5_eigenvalues_MHz, 3).tolist()}")
    map_polar_angles_rad = np.radians(np.arange(0.0, 180.0 + 1e-9, SCAN_STEP_DEG))
    map_azimuths_rad = np.radians(np.arange(-180.0, 180.0 - 1e-9, SCAN_STEP_DEG))
    grid_polar, grid_azimuth = np.meshgrid(map_polar_angles_rad, map_azimuths_rad, indexing="ij")
    scan_polar_angles_rad, scan_azimuths_rad = grid_polar.ravel(), grid_azimuth.ravel()  # θ slowest
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

    FORMAT_VERSION = 2  # version 1 scanned 590 Lebedev directions instead of this grid
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
