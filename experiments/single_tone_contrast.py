import os

# one BLAS thread per process; has to come before numpy
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from control import Drive, drive_rad_per_us
from engine import field_direction_yields, recombination_yield
from model import flavin_tryptophan_hypothesis
from results import format_duration, new_run_id, save_quick_look, save_run


# %% settings
# new
# pairs, directions, cap, window and step all as in optimise_drive.py, so the two can be compared
FIELD_STRENGTH_uT = 50.0
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)
CONTRAST_FIELD_DIRECTIONS = (np.array([1.4393, 0.4242, 2.6949, 1.3668]),
                             np.array([-2.6991, 3.0531, -0.5672, 3.127]))

AMPLITUDE_CAP_MHz = 0.25 * HYPOTHESIS_A.zeeman_frequency_MHz
TONE_FREQUENCIES_MHz = np.arange(0.5, 60.01, 0.5)
TIMES_US = np.linspace(0, 12, 4001)
MAX_STEP_US = 0.00151

NUMBER_OF_PROCESSES = 4


# %% functions
# new
def direction_yields(frequency_MHz):
    # one sine at the cap with phase 0 (None = no drive); returns recombination yields, shape (pair, direction)
    drive_function = None if frequency_MHz is None else drive_rad_per_us(
        Drive(np.array([frequency_MHz]), np.array([AMPLITUDE_CAP_MHz]), np.array([0.0])))
    yields_by_pair = []
    for hypothesis in (HYPOTHESIS_A, HYPOTHESIS_B):
        singlet_yields, _ = field_direction_yields(hypothesis, drive_function, TIMES_US, MAX_STEP_US,
                                                   CONTRAST_FIELD_DIRECTIONS)
        yields_by_pair.append(recombination_yield(hypothesis, singlet_yields))
    return np.array(yields_by_pair)


# new
def contrasts(yields_by_pair):
    return np.max(yields_by_pair, axis=-1) - np.min(yields_by_pair, axis=-1)


# %% run
# new
if __name__ == "__main__":
    start = time.perf_counter()
    static_yields = direction_yields(None)
    if NUMBER_OF_PROCESSES == 1:
        tone_yields = np.array([direction_yields(frequency_MHz) for frequency_MHz in TONE_FREQUENCIES_MHz])
    else:
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as pool:
            tone_yields = np.array(list(pool.map(direction_yields, TONE_FREQUENCIES_MHz)))
    print(f"{len(TONE_FREQUENCIES_MHz)} tones in {format_duration(time.perf_counter() - start)}")

    static_contrast_A, static_contrast_B = contrasts(static_yields)
    tone_contrasts = contrasts(tone_yields)
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
