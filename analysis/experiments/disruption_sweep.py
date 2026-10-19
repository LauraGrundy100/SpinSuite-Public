"""Disruption across hyperfine anisotropies: static yield, best single-frequency drive, and optimised drive.

For each anisotropy δ: the static singlet yield, a scan of single-frequency drives (amplitude = cap, phase 0),
action-spectrum starting frequencies, and (if SUPER_ITERATIONS > 0) the dCRAB-inspired search. Each δ runs in its
own process, side by side. Each search saves a checkpoint after every super-iteration; starting the script again
with the same settings carries on where it stopped.
Side-by-side runs must be started from the terminal (python experiments/disruption_sweep.py);
for the VS Code interactive window set NUMBER_OF_PROCESSES = 1.
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

# Make model.py, engine.py and control.py importable, whether run as a script or cell by cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from control import (Drive, action_spectrum, action_spectrum_seeds, disruption_cost, drive_rad_per_us, l2_drive,
                     linf_drive, optimise_drive)
from engine import yields
from model import baseline_hypothesis
from results import finish_run, save_quick_look, save_run, start_or_resume_run

# %% Settings

ANISOTROPIES = [0.0, 0.1, 0.2, 0.3, 0.4]  # hyperfine anisotropy δ values (dimensionless); APEX's values
NUMBER_OF_PROCESSES = 5  # side-by-side processes (this laptop has 12 cores); 1 = one after another

AMPLITUDE_CAP_MHz = 7.0  # strength cap: 0.25 × A_iso, DISCO value (may be stale)
CAP_METHOD = "peak"  # "peak" (L-infinity: highest point = cap) or "power" (L2: √(Σ a²) = cap)
PEAK_CHECK_TIMES_US = np.linspace(0, 12, 20000)  # where the peak cap looks; this is the old repo's grid

TIMES_US = np.linspace(0, 10, 3000)  # yield window and output grid (µs); APEX's grid
MAX_STEP_US = 0.0015  # largest simulation step (µs). On this grid: 3 steps of 1.11 ns per output interval. The next
#                      option, 2 steps of 1.67 ns (MAX_STEP_US >= 0.00167), is faster but its accuracy is not measured

MONOCHROMATIC_FREQUENCIES_MHz = np.linspace(0.1, 40, 300)  # single-frequency scan; APEX's grid

NUMBER_OF_FREQUENCIES = 4  # number of frequencies in the drive
SEED_BIN_WIDTH_MHz = 1.0  # action-spectrum bin width
SEED_MAX_FREQUENCY_MHz = 40.0  # highest action-spectrum bin edge
SEED_MIN_SEPARATION_MHz = 0.5  # action-spectrum starting frequencies closer than this are skipped

SUPER_ITERATIONS = 14  # most rounds of the search loop (APEX: 14); 0 = no search, only static, scan and seeds
PATIENCE_SUPER_ITERATIONS = 6  # stop after this many rounds in a row without improvement (= SUPER_ITERATIONS: never)
FREQUENCY_PERTURBATION_MHz = 5.0  # each round nudges every frequency by up to ± this
MAX_NELDER_MEAD_ITERATIONS = 1000  # cap on fine-tuning steps per round (old value 1000)
RANDOM_SEED = 123  # fixes the random nudges; 123 reproduces the old runs' nudges

########################################################################################################
########################################################################################################

# %% One anisotropy (runs in its own process)


def make_drive(frequencies_MHz, raw_amplitudes, phases_rad):
    """Apply the strength cap chosen in the settings."""
    if CAP_METHOD == "peak":
        return linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz, PEAK_CHECK_TIMES_US)
    if CAP_METHOD == "power":
        return l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz)
    raise ValueError(f"CAP_METHOD must be 'peak' or 'power', got {CAP_METHOD!r}")


def run_one_anisotropy(delta, checkpoint_folder):
    """Static yield, single-frequency scan, starting frequencies and (optionally) the search for one δ, the search
    checkpointed in checkpoint_folder. Returns a dict of results. On a resume only the search carries on; the
    static yield and the scan (about a minute) are recalculated."""
    start_time = time.perf_counter()
    hypothesis = baseline_hypothesis(delta)  # DISCO parameters, may be stale placeholders
    static_yield, _ = yields(hypothesis, None, TIMES_US, MAX_STEP_US)
    monochromatic_yields = np.array([
        disruption_cost(hypothesis, Drive(np.array([frequency_MHz]), np.array([AMPLITUDE_CAP_MHz]), np.array([0.0])),
                        TIMES_US, MAX_STEP_US)
        for frequency_MHz in MONOCHROMATIC_FREQUENCIES_MHz])
    print(f"[delta = {delta}] static and single-frequency scan done, {time.perf_counter() - start_time:.0f} s")

    centres_MHz, scores = action_spectrum(hypothesis, SEED_BIN_WIDTH_MHz, SEED_MAX_FREQUENCY_MHz)
    starting_frequencies_MHz = action_spectrum_seeds(centres_MHz, scores, NUMBER_OF_FREQUENCIES, SEED_MIN_SEPARATION_MHz)

    def cost(frequencies_MHz, raw_amplitudes, phases_rad):
        return disruption_cost(hypothesis, make_drive(frequencies_MHz, raw_amplitudes, phases_rad), TIMES_US,
                               MAX_STEP_US)

    result = optimise_drive(cost, starting_frequencies_MHz, SUPER_ITERATIONS, FREQUENCY_PERTURBATION_MHz,
                            MAX_NELDER_MEAD_ITERATIONS, RANDOM_SEED, PATIENCE_SUPER_ITERATIONS,
                            checkpoint_path=checkpoint_folder / f"delta_{delta}.npz")
    best_drive = make_drive(result.frequencies_MHz, result.raw_amplitudes, result.phases_rad)
    print(f"[delta = {delta}] finished, {time.perf_counter() - start_time:.0f} s")
    return {"static_yield": static_yield, "monochromatic_yields": monochromatic_yields, "starting_frequencies_MHz": starting_frequencies_MHz,
            "frequencies_MHz": result.frequencies_MHz, "raw_amplitudes": result.raw_amplitudes,
            "phases_rad": result.phases_rad, "amplitudes_MHz": best_drive.amplitudes_MHz,
            "cost_history": result.cost_history, "accepted": result.accepted, "evaluations": result.evaluations}


# %% Run every anisotropy, print a summary, save

if __name__ == "__main__":
    # Every setting is saved with the run. A run with exactly these settings that was interrupted is resumed.
    FORMAT_VERSION = 2  # 2: checkpoints; number_of_seeds renamed number_of_frequencies, seeds_MHz renamed
    #                        starting_frequencies_MHz
    settings = dict(
        anisotropies=ANISOTROPIES, amplitude_cap_MHz=AMPLITUDE_CAP_MHz, cap_method=CAP_METHOD,
        peak_check_times_us=PEAK_CHECK_TIMES_US, times_us=TIMES_US, max_step_us=MAX_STEP_US,
        monochromatic_frequencies_MHz=MONOCHROMATIC_FREQUENCIES_MHz, number_of_frequencies=NUMBER_OF_FREQUENCIES,
        seed_bin_width_MHz=SEED_BIN_WIDTH_MHz, seed_max_frequency_MHz=SEED_MAX_FREQUENCY_MHz,
        seed_min_separation_MHz=SEED_MIN_SEPARATION_MHz, super_iterations=SUPER_ITERATIONS,
        patience_super_iterations=PATIENCE_SUPER_ITERATIONS, frequency_perturbation_MHz=FREQUENCY_PERTURBATION_MHz,
        max_nelder_mead_iterations=MAX_NELDER_MEAD_ITERATIONS, random_seed=RANDOM_SEED)
    results_folder = REPO_FOLDER / "results"
    run_id, run_checkpoint_folder = start_or_resume_run(results_folder, "disruption_sweep", settings)

    sweep_start = time.perf_counter()
    if NUMBER_OF_PROCESSES == 1:
        outcomes = [run_one_anisotropy(delta, run_checkpoint_folder) for delta in ANISOTROPIES]
    else:
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as pool:
            outcomes = list(pool.map(run_one_anisotropy, ANISOTROPIES, [run_checkpoint_folder] * len(ANISOTROPIES)))
    print(f"Sweep finished in {time.perf_counter() - sweep_start:.0f} s")

    print("  delta static   best single (MHz)   single supp.   optimised   optimised supp.")
    for delta, outcome in zip(ANISOTROPIES, outcomes):
        best = np.argmin(outcome["monochromatic_yields"])
        single_yield, optimised_yield = outcome["monochromatic_yields"][best], outcome["cost_history"][-1]
        print(f"  {delta:.1f}  {outcome['static_yield']:.6f}  {single_yield:.6f} ({MONOCHROMATIC_FREQUENCIES_MHz[best]:5.2f})"
              f"   {100 * (1 - single_yield / outcome['static_yield']):5.1f}%       {optimised_yield:.6f}"
              f"   {100 * (1 - optimised_yield / outcome['static_yield']):5.1f}%")

    per_anisotropy = {f"delta_{delta}_{key}": value for delta, outcome in zip(ANISOTROPIES, outcomes)
                      for key, value in outcome.items()}  # outcome, one set of names per anisotropy
    save_run(results_folder, run_id, FORMAT_VERSION, settings, per_anisotropy)
    save_quick_look(results_folder, run_id, "scan", "drive frequency (MHz)", "singlet yield",
                    [(MONOCHROMATIC_FREQUENCIES_MHz, outcome["monochromatic_yields"], f"delta = {delta}")
                     for delta, outcome in zip(ANISOTROPIES, outcomes)])
    save_quick_look(results_folder, run_id, "cost", "super-iteration (0 = start)", "best singlet yield so far",
                    [(np.arange(len(outcome["cost_history"])), outcome["cost_history"], f"delta = {delta}")
                     for delta, outcome in zip(ANISOTROPIES, outcomes)])
    finish_run(results_folder, run_id)  # label its checkpoints finished; keep only the newest three finished runs'
    print(f"Saved results/{run_id}.npz, results/{run_id}_scan.png and results/{run_id}_cost.png")
