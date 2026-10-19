"""Single-tone benchmark for the contrast: each pair's contrast under one sine drive at the cap, scanned over
frequency, next to the contrast with no drive.

Contrast = highest minus lowest recombination yield over CONTRAST_FIELD_DIRECTIONS (engine.recombination_contrast).
Deterministic (no random numbers). A yardstick for optimised drives (IDEAS.md idea 4 style). Frequencies are shared
out between NUMBER_OF_PROCESSES processes, which must be started from the terminal
(python experiments/single_tone_contrast.py); set NUMBER_OF_PROCESSES = 1 in the interactive window.
"""

import os

# One thread per process, so side-by-side processes do not compete for cores. Must come before importing NumPy.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Make model.py, engine.py, control.py and results.py importable, whether run as a script or cell by cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from control import Drive, drive_rad_per_us
from engine import field_direction_yields, recombination_yield
from model import flavin_tryptophan_hypothesis
from results import format_duration, new_run_id, save_quick_look, save_run

# %% Settings

FIELD_STRENGTH_uT = 50.0  # geomagnetic field strength (µT). Assumption: Daniel gave no strength
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)  # classical: chi 0, k_r = k_f = 1 per µs
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)  # CISS: chi pi/2, k_r 10, k_f 1
# Field directions for the contrast, (polar angles, azimuths) in rad in the molecule's frame: both pairs' lowest and
# highest from field_scan_2026-10-15_01. Keep the same as CONTRAST_FIELD_DIRECTIONS in optimise_drive.py.
CONTRAST_FIELD_DIRECTIONS = (np.array([1.4393, 0.4242, 2.6949, 1.3668]),  # pair A lowest, A highest,
                             np.array([-2.6991, 3.0531, -0.5672, 3.127]))  # B lowest, B highest

AMPLITUDE_CAP_MHz = 0.25 * HYPOTHESIS_A.zeeman_frequency_MHz  # placeholder cap, as in optimise_drive.py (~0.35 MHz)
TONE_FREQUENCIES_MHz = np.arange(0.5, 60.01, 0.5)  # single-tone frequencies (MHz); FN5 reaches about 49 MHz
TIMES_US = np.linspace(0, 12, 4001)  # yield window and output grid (µs), as in optimise_drive.py
MAX_STEP_US = 0.00151  # largest simulation step (µs), as in optimise_drive.py: 2 steps of 1.5 ns per interval

NUMBER_OF_PROCESSES = 4  # side by side (this laptop: 6 cores; 4 leaves room for other work); 1 = one after another

########################################################################################################
########################################################################################################

# %% Recombination yields in every contrast direction, for one drive

def direction_yields(frequency_MHz):
    """Recombination yields of pair A and pair B, one per contrast direction, under one sine at the cap
    (amplitude = cap, phase 0) at frequency_MHz; None gives no drive."""
    drive_function = None if frequency_MHz is None else drive_rad_per_us(
        Drive(np.array([frequency_MHz]), np.array([AMPLITUDE_CAP_MHz]), np.array([0.0])))
    yields_by_pair = []
    for hypothesis in (HYPOTHESIS_A, HYPOTHESIS_B):
        singlet_yields, _ = field_direction_yields(hypothesis, drive_function, TIMES_US, MAX_STEP_US,
                                                   CONTRAST_FIELD_DIRECTIONS)
        yields_by_pair.append(recombination_yield(hypothesis, singlet_yields))
    return np.array(yields_by_pair)  # shape (pair, direction)


def contrasts(yields_by_pair):
    """(C_A, C_B): highest minus lowest recombination yield over the directions, for each pair."""
    return np.max(yields_by_pair, axis=-1) - np.min(yields_by_pair, axis=-1)


# %% Scan, print, save

if __name__ == "__main__":
    start = time.perf_counter()
    static_yields = direction_yields(None)
    if NUMBER_OF_PROCESSES == 1:
        tone_yields = np.array([direction_yields(frequency_MHz) for frequency_MHz in TONE_FREQUENCIES_MHz])
    else:
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as pool:
            tone_yields = np.array(list(pool.map(direction_yields, TONE_FREQUENCIES_MHz)))  # (frequency, pair, direction)
    print(f"{len(TONE_FREQUENCIES_MHz)} tones in {format_duration(time.perf_counter() - start)}")

    static_contrast_A, static_contrast_B = contrasts(static_yields)
    tone_contrasts = contrasts(tone_yields)  # shape (frequency, pair)
    tone_differences = np.abs(tone_contrasts[:, 0] - tone_contrasts[:, 1])
    best = np.argmax(tone_differences)
    print(f"Cap {AMPLITUDE_CAP_MHz:.4f} MHz. No drive: C_A {static_contrast_A:.6f}, C_B {static_contrast_B:.6f}, "
          f"|C_A - C_B| {abs(static_contrast_A - static_contrast_B):.6f}")
    print(f"Best single tone for |C_A - C_B|: {TONE_FREQUENCIES_MHz[best]:.1f} MHz, C_A {tone_contrasts[best, 0]:.6f}, "
          f"C_B {tone_contrasts[best, 1]:.6f}, |C_A - C_B| {tone_differences[best]:.6f}")
    for label, column in (("A", 0), ("B", 1)):
        lowest, highest = np.argmin(tone_contrasts[:, column]), np.argmax(tone_contrasts[:, column])
        print(f"  C_{label} ranges from {tone_contrasts[lowest, column]:.6f} ({TONE_FREQUENCIES_MHz[lowest]:.1f} MHz) to "
              f"{tone_contrasts[highest, column]:.6f} ({TONE_FREQUENCIES_MHz[highest]:.1f} MHz)")

    FORMAT_VERSION = 1
    settings = dict(hypothesis_A=repr(HYPOTHESIS_A), hypothesis_B=repr(HYPOTHESIS_B),
                    field_strength_uT=FIELD_STRENGTH_uT, contrast_polar_angles_rad=CONTRAST_FIELD_DIRECTIONS[0],
                    contrast_azimuths_rad=CONTRAST_FIELD_DIRECTIONS[1], amplitude_cap_MHz=AMPLITUDE_CAP_MHz,
                    tone_frequencies_MHz=TONE_FREQUENCIES_MHz, times_us=TIMES_US, max_step_us=MAX_STEP_US)
    outcome = dict(static_direction_yields=static_yields, tone_direction_yields=tone_yields,
                   static_contrast_A=static_contrast_A, static_contrast_B=static_contrast_B,
                   tone_contrasts_A=tone_contrasts[:, 0], tone_contrasts_B=tone_contrasts[:, 1])
    results_folder = REPO_FOLDER / "results"
    run_id = new_run_id(results_folder, "single_tone_contrast")
    save_run(results_folder, run_id, FORMAT_VERSION, settings, outcome)
    save_quick_look(results_folder, run_id, "contrast", "tone frequency (MHz)", "contrast (highest - lowest yield)",
                    [(TONE_FREQUENCIES_MHz, tone_contrasts[:, 0], "pair A"), (TONE_FREQUENCIES_MHz, tone_contrasts[:, 1],
                                                                              "pair B")])
    print(f"Saved results/{run_id}.npz and results/{run_id}_contrast.png")
