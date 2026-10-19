"""Null test: is an optimised drive better than luck? Scores many random, unoptimised drives of the same strength
and compares them with the best drive of a saved run.

Each random drive has as many frequencies as the optimised one, uniform frequencies, amplitudes and phases, and is
scored once. The cap, cap method, time grid, simulation step and orientations (the run's final directions; none for
runs before 2026-10-13) are read from the saved run, so the comparison is like for like, and the optimised drive is
re-scored here the same way. Reported: how many random drives beat the
optimised drive by the run's own measure. Side-by-side processes must be started from the terminal
(python experiments/null_test.py); in the VS Code interactive window set NUMBER_OF_PROCESSES = 1.
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

from control import drive_rad_per_us, l2_drive, linf_drive, random_drive_parameters
from engine import bhattacharyya_distance, orientation_grid, outcome_probabilities, pairs_needed, yields_together
from model import baseline_hypothesis, ciss_density_matrix
from results import format_duration, load_run, new_run_id, progress_line, save_quick_look, save_run

# %% Settings

OPTIMISED_RUN_ID = "discrimination_2026-10-14_01"  # run whose best drive is tested (results/<run ID>.npz);
#                                                    this one is the exploratory six-seed run averaged over
#                                                    orientations, singlet-born vs CISS-born
# The pairs the optimised run used; checked against the descriptions saved with it.
# Baseline pairs use DISCO's parameters (old repo), which may be stale placeholders.
# For runs up to 2026-10-08 (anisotropy 0.0 vs 0.3) use baseline_hypothesis(0.0) and baseline_hypothesis(0.3).
HYPOTHESIS_A = baseline_hypothesis(0.3)  # singlet-born
HYPOTHESIS_B = baseline_hypothesis(0.3).variant(  # the same pair, but CISS-tilted at the start (chi = 0.1 pi)
    name="baseline, delta=0.3, CISS chi=0.1pi", electron_initial_state=ciss_density_matrix(0.1 * np.pi))

NUMBER_OF_RANDOM_DRIVES = 1000  # if none beats the optimised drive, fewer than 3/1000 would (95% confidence).
#                                 Averaged runs take longer per drive (194 directions: about 7 s, against 0.3 s)
RANDOM_MAX_FREQUENCY_MHz = 40.0  # random frequencies uniform in 0 to this (MHz); 40 is the action-spectrum window.
#                                  Up to 100 is safe: engine accuracy measured to 100 MHz (2026-10-08)
RANDOM_SEED = 1  # fixes the whole set of random drives
ERROR_PROBABILITY = 0.01  # for the printed "pairs needed": pairs to tell A from B with at most this chance of error
NUMBER_OF_PROCESSES = 6  # side-by-side processes, each scoring a share of the drives; 1 = one after another.
#                          Results do not depend on it

########################################################################################################
########################################################################################################

# %% Scoring drives


def make_drive(frequencies_MHz, raw_amplitudes, phases_rad, run):
    """Apply the saved run's strength cap to a drive."""
    if run["cap_method"] == "peak":
        return linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, run["amplitude_cap_MHz"],
                          run["peak_check_times_us"])
    return l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, run["amplitude_cap_MHz"])


def yields_of_drives(drives, run, label):
    """Yields (Y_S,A, Y_T,A, Y_S,B, Y_T,B), one row per drive, for a list of (frequencies, raw amplitudes, phases)
    under the saved run's cap, time grid, step and final orientations. None in the list means no drive.
    Prints progress every 50."""
    orientations = None if run["final_lebedev_order"] == 0 else orientation_grid(run["final_lebedev_order"])
    start = time.perf_counter()
    rows = np.zeros((len(drives), 4))
    for index, parameters in enumerate(drives):
        drive_function = None if parameters is None else drive_rad_per_us(make_drive(*parameters, run))
        (singlet_A, triplet_A), (singlet_B, triplet_B) = yields_together(
            [HYPOTHESIS_A, HYPOTHESIS_B], drive_function, run["times_us"], run["max_step_us"], orientations)
        rows[index] = singlet_A, triplet_A, singlet_B, triplet_B
        if (index + 1) % 50 == 0:
            print(f"  [{label}] {progress_line('drive', index + 1, len(drives), time.perf_counter() - start)}",
                  flush=True)
    return rows


def scores(yield_rows, run):
    """The run's own measure for each row of yields, higher is better: −Y_S,A (disruption), the gap D, or √B."""
    if run["goal"] == "disruption":
        return -yield_rows[:, 0]
    if run["discrimination_measure"] == "gap":
        return np.abs(yield_rows[:, 0] - yield_rows[:, 2])
    return np.sqrt([bhattacharyya_distance(outcome_probabilities(*row[:2]), outcome_probabilities(*row[2:]))
                    for row in yield_rows])


def describe(row):
    """Y_S,A, gap D and pairs needed for one row of yields, as text."""
    bhattacharyya = bhattacharyya_distance(outcome_probabilities(*row[:2]), outcome_probabilities(*row[2:]))
    return (f"Y_S,A {row[0]:.6f}, gap D {abs(row[0] - row[2]):.6f}, "
            f"pairs needed {pairs_needed(bhattacharyya, ERROR_PROBABILITY):.0f}")


# %% Draw the random drives, score them and the optimised drive, compare, save

if __name__ == "__main__":
    run = load_run(REPO_FOLDER / "results" / f"{OPTIMISED_RUN_ID}.npz")
    if run["hypothesis_A"] != repr(HYPOTHESIS_A) or run["hypothesis_B"] != repr(HYPOTHESIS_B):
        raise ValueError(f"HYPOTHESIS_A and HYPOTHESIS_B must be the pairs {OPTIMISED_RUN_ID} used")
    measure = "Y_S,A (lower is better)" if run["goal"] == "disruption" else {
        "gap": "gap D", "bhattacharyya": "sqrt(B)"}[run["discrimination_measure"]]

    # Every drive is drawn here, from one generator, so the set does not depend on the number of processes.
    random_numbers = np.random.default_rng(RANDOM_SEED)
    random_drives = [random_drive_parameters(random_numbers, run["number_of_frequencies"], RANDOM_MAX_FREQUENCY_MHz)
                     for _ in range(NUMBER_OF_RANDOM_DRIVES)]
    start = time.perf_counter()
    if NUMBER_OF_PROCESSES == 1:
        random_yields = yields_of_drives(random_drives, run, "random drives")
    else:
        share_size = -(-NUMBER_OF_RANDOM_DRIVES // NUMBER_OF_PROCESSES)  # divide, rounding up
        shares = [random_drives[first:first + share_size] for first in range(0, NUMBER_OF_RANDOM_DRIVES, share_size)]
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as pool:
            random_yields = np.concatenate(list(pool.map(  # map keeps the shares in order
                yields_of_drives, shares, [run] * len(shares), [f"share {number + 1}" for number in range(len(shares))])))
    optimised_drive = (run["frequencies_MHz"], run["raw_amplitudes"], run["phases_rad"])
    optimised_yields, static_yields = yields_of_drives([optimised_drive, None], run, "optimised and no drive")
    print(f"Scored {NUMBER_OF_RANDOM_DRIVES} random drives in {format_duration(time.perf_counter() - start)}, "
          + ("molecules as given" if run["final_lebedev_order"] == 0 else
             f"averaged over {len(orientation_grid(run['final_lebedev_order'])[2])} orientations"))

    random_scores, optimised_score = scores(random_yields, run), scores(optimised_yields[None, :], run)[0]
    number_beating = int(np.sum(random_scores > optimised_score))
    best_seed_note = f", random seed {run['best_random_seed']}" if "best_random_seed" in run else ""
    print(f"Optimised drive ({OPTIMISED_RUN_ID}{best_seed_note}): {describe(optimised_yields)}")
    print(f"No drive:                {describe(static_yields)}")
    print(f"Random drives ({NUMBER_OF_RANDOM_DRIVES}, {run['number_of_frequencies']} frequencies in "
          f"0-{RANDOM_MAX_FREQUENCY_MHz:g} MHz, same cap):")
    print(f"  median: {describe(random_yields[np.argsort(random_scores)[len(random_scores) // 2]])}")
    print(f"  best:   {describe(random_yields[np.argmax(random_scores)])}")
    if number_beating == 0:  # rule of three: if none of N tries succeeds, the chance is very likely below 3/N
        print(f"Beating the optimised drive by {measure}: none of {NUMBER_OF_RANDOM_DRIVES}, so very likely fewer than "
              f"{100 * 3 / NUMBER_OF_RANDOM_DRIVES:.1f}% of random drives would (rule of three, 95% confidence)")
    else:
        print(f"Beating the optimised drive by {measure}: {number_beating} of {NUMBER_OF_RANDOM_DRIVES} "
              f"({100 * number_beating / NUMBER_OF_RANDOM_DRIVES:.1f}%)")

    FORMAT_VERSION = 2  # 2: adds final_lebedev_order (version 1 had no averaging and is read as 0)
    settings = dict(
        optimised_run_id=OPTIMISED_RUN_ID, hypothesis_A=repr(HYPOTHESIS_A), hypothesis_B=repr(HYPOTHESIS_B),
        number_of_random_drives=NUMBER_OF_RANDOM_DRIVES, random_max_frequency_MHz=RANDOM_MAX_FREQUENCY_MHz,
        random_seed=RANDOM_SEED, error_probability=ERROR_PROBABILITY,
        # read from the optimised run, so the comparison is like for like
        goal=run["goal"], discrimination_measure=run.get("discrimination_measure", ""),
        number_of_frequencies=run["number_of_frequencies"], amplitude_cap_MHz=run["amplitude_cap_MHz"],
        cap_method=run["cap_method"], peak_check_times_us=run["peak_check_times_us"], times_us=run["times_us"],
        max_step_us=run["max_step_us"], final_lebedev_order=run["final_lebedev_order"])
    outcome = dict(  # yields columns: Y_S,A, Y_T,A, Y_S,B, Y_T,B
        random_frequencies_MHz=np.array([drive[0] for drive in random_drives]),
        random_raw_amplitudes=np.array([drive[1] for drive in random_drives]),
        random_phases_rad=np.array([drive[2] for drive in random_drives]),
        random_yields=random_yields, optimised_yields=optimised_yields, static_yields=static_yields,
        measure=measure, random_scores=random_scores, optimised_score=optimised_score, number_beating=number_beating)
    results_folder = REPO_FOLDER / "results"
    run_id = new_run_id(results_folder, "null_test")
    save_run(results_folder, run_id, FORMAT_VERSION, settings, outcome)
    sign = -1 if run["goal"] == "disruption" else 1  # plot Y_S,A itself for disruption
    fraction = (np.arange(NUMBER_OF_RANDOM_DRIVES) + 0.5) / NUMBER_OF_RANDOM_DRIVES
    save_quick_look(results_folder, run_id, "scores", "random drives, sorted from worst to best (fraction)",
                    measure.replace(" (lower is better)", ""),
                    [(fraction, sign * np.sort(random_scores), "random drives"),
                     ([0, 1], [sign * optimised_score] * 2, "optimised drive"),
                     ([0, 1], [sign * scores(static_yields[None, :], run)[0]] * 2, "no drive")])
    print(f"Saved results/{run_id}.npz and results/{run_id}_scores.png")
