import os

# one BLAS thread per process, so the side-by-side seeds don't fight over cores; has to come before numpy
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from control import (action_spectrum, action_spectrum_seeds, bhattacharyya_cost, contrast_cost, discrimination_cost,
                     disruption_cost, drive_rad_per_us, l2_drive, linf_drive, optimise_drive,
                     random_starting_frequencies)
from engine import (bhattacharyya_distance, orientation_grid, outcome_probabilities, pairs_needed,
                    recombination_contrast, yields_together)
from model import field_direction_hypothesis, flavin_tryptophan_hypothesis
from results import finish_run, format_duration, save_quick_look, save_run, start_or_resume_run


# %% settings
# new
GOAL = "discrimination"
DISCRIMINATION_MEASURE = "contrast"  # objective: maximise |C_A - C_B| ("gap" and "bhattacharyya" are the older ones)
ERROR_PROBABILITY = 0.01
FIELD_STRENGTH_uT = 50.0  # my assumption, Daniel gave no field strength
# rates and χ from Daniel: classical k_r = k_f = 1 µs⁻¹, χ = 0; CISS χ = π/2 (maximal), k_r = 10, k_f = 1 µs⁻¹
# check: k_r = 10 for CISS is what Daniel says is usually assumed; I don't have the physical reason written down
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)
# (θ, φ) in rad, molecule frame: lowest and highest yield of A, then of B, from the static scan field_scan_2026-10-15_01
CONTRAST_FIELD_DIRECTIONS = (np.array([1.4393, 0.4242, 2.6949, 1.3668]),
                             np.array([-2.6991, 3.0531, -0.5672, 3.127]))

AMPLITUDE_CAP_MHz = 0.25 * HYPOTHESIS_A.zeeman_frequency_MHz  # placeholder, about 0.35 MHz
CAP_METHOD = "peak"
PEAK_CHECK_TIMES_US = np.linspace(0, 12, 20000)

TIMES_US = np.linspace(0, 12, 4001)  # 3 ns grid; 12 µs is about 12 lifetimes of 1/k_f (Daniel: 10 to 12)
MAX_STEP_US = 0.00151  # 2 steps of 1.5 ns per grid interval

SEARCH_LEBEDEV_ORDER = 0  # 0 = no orientation averaging; must stay 0 for the contrast
FINAL_LEBEDEV_ORDER = 0

NUMBER_OF_FREQUENCIES = 4
STARTING_FREQUENCIES_FROM = "action spectrum"
RANDOM_START_MAX_FREQUENCY_MHz = 40.0
SEED_BIN_WIDTH_MHz = 1.0
SEED_MAX_FREQUENCY_MHz = 60.0  # FN5 couplings reach about 49 MHz
SEED_MIN_SEPARATION_MHz = 2.0

SUPER_ITERATIONS = 5
PATIENCE_SUPER_ITERATIONS = 5  # equal to SUPER_ITERATIONS, so it never stops early
FREQUENCY_PERTURBATION_MHz = 5.0
MAX_NELDER_MEAD_ITERATIONS = 1000
RANDOM_SEEDS = [1, 2, 3, 4, 5]
NUMBER_OF_PROCESSES = 5  # doesn't change results, so it isn't saved


# %% functions
# new
def orientations(lebedev_order):
    return None if lebedev_order == 0 else orientation_grid(lebedev_order)


SEARCH_ORIENTATIONS = orientations(SEARCH_LEBEDEV_ORDER)
FINAL_ORIENTATIONS = orientations(FINAL_LEBEDEV_ORDER)


# old, edited: maximum also taken over the contrast directions
def action_spectrum_starting_frequencies():
    # bin-by-bin maximum of the action spectra of both pairs, each with the field along every contrast direction
    if DISCRIMINATION_MEASURE == "contrast":
        scored = [field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
                  for hypothesis in (HYPOTHESIS_A, HYPOTHESIS_B)
                  for polar_angle_rad, azimuth_rad in zip(*CONTRAST_FIELD_DIRECTIONS)]
    else:
        scored = [HYPOTHESIS_A, HYPOTHESIS_B] if GOAL == "discrimination" else [HYPOTHESIS_A]
    spectra = [action_spectrum(hypothesis, SEED_BIN_WIDTH_MHz, SEED_MAX_FREQUENCY_MHz, SEARCH_ORIENTATIONS)
               for hypothesis in scored]
    centres_MHz, scores = spectra[0][0], np.max([spectrum_scores for _, spectrum_scores in spectra], axis=0)
    return action_spectrum_seeds(centres_MHz, scores, NUMBER_OF_FREQUENCIES, SEED_MIN_SEPARATION_MHz)


# new
def make_drive(frequencies_MHz, raw_amplitudes, phases_rad):
    if CAP_METHOD == "peak":
        return linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz, PEAK_CHECK_TIMES_US)
    return l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz)


# new
def cost(frequencies_MHz, raw_amplitudes, phases_rad):
    drive = make_drive(frequencies_MHz, raw_amplitudes, phases_rad)
    if GOAL == "disruption":
        return disruption_cost(HYPOTHESIS_A, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)
    if DISCRIMINATION_MEASURE == "gap":
        return discrimination_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)
    if DISCRIMINATION_MEASURE == "contrast":
        return contrast_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, CONTRAST_FIELD_DIRECTIONS)
    return bhattacharyya_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)


# new
def both_yields(drive):
    return yields_together([HYPOTHESIS_A, HYPOTHESIS_B], None if drive is None else drive_rad_per_us(drive),
                           TIMES_US, MAX_STEP_US, FINAL_ORIENTATIONS)


# new
def both_contrasts(drive):
    drive_function = None if drive is None else drive_rad_per_us(drive)
    return tuple(recombination_contrast(hypothesis, drive_function, TIMES_US, MAX_STEP_US, CONTRAST_FIELD_DIRECTIONS)
                 for hypothesis in (HYPOTHESIS_A, HYPOTHESIS_B))


# new
def final_cost(yields_A, yields_B):
    # picks the best seed; for the contrast it gets (C_A,), (C_B,) and is just the search's own cost again
    if GOAL == "disruption":
        return yields_A[0]
    if DISCRIMINATION_MEASURE in ("gap", "contrast"):
        return -abs(yields_A[0] - yields_B[0])
    return -np.sqrt(bhattacharyya_of(yields_A, yields_B))


# new
def run_one_seed(random_seed, checkpoint_folder):
    if STARTING_FREQUENCIES_FROM == "random":
        starting_frequencies_MHz = random_starting_frequencies(random_seed, NUMBER_OF_FREQUENCIES,
                                                               RANDOM_START_MAX_FREQUENCY_MHz)
    else:
        starting_frequencies_MHz = action_spectrum_starting_frequencies()
    result = optimise_drive(cost, starting_frequencies_MHz, SUPER_ITERATIONS, FREQUENCY_PERTURBATION_MHz,
                            MAX_NELDER_MEAD_ITERATIONS, random_seed, PATIENCE_SUPER_ITERATIONS,
                            checkpoint_path=checkpoint_folder / f"random_seed_{random_seed}.npz")
    best_drive = make_drive(result.frequencies_MHz, result.raw_amplitudes, result.phases_rad)
    outcome = {"starting_frequencies_MHz": starting_frequencies_MHz, "frequencies_MHz": result.frequencies_MHz,
               "raw_amplitudes": result.raw_amplitudes, "phases_rad": result.phases_rad,
               "amplitudes_MHz": best_drive.amplitudes_MHz, "cost_history": result.cost_history,
               "accepted": result.accepted, "evaluations": result.evaluations,
               "nelder_mead_costs": result.nelder_mead_costs}
    if DISCRIMINATION_MEASURE == "contrast":
        contrast_A, contrast_B = both_contrasts(best_drive)
        outcome.update(driven_contrast_A=contrast_A, driven_contrast_B=contrast_B,
                       final_cost=final_cost((contrast_A,), (contrast_B,)))
        return outcome
    (yield_A, triplet_yield_A), (yield_B, triplet_yield_B) = both_yields(best_drive)
    outcome.update(driven_yield_A=yield_A, driven_triplet_yield_A=triplet_yield_A,
                   driven_yield_B=yield_B, driven_triplet_yield_B=triplet_yield_B,
                   final_cost=final_cost((yield_A, triplet_yield_A), (yield_B, triplet_yield_B)))
    return outcome


# new
def bhattacharyya_of(yields_A, yields_B):
    return bhattacharyya_distance(outcome_probabilities(*yields_A), outcome_probabilities(*yields_B))


# new
def print_yield_summary(outcomes):
    (static_yield_A, static_triplet_yield_A), (static_yield_B, static_triplet_yield_B) = both_yields(None)
    static_bhattacharyya = bhattacharyya_of((static_yield_A, static_triplet_yield_A),
                                            (static_yield_B, static_triplet_yield_B))
    print(f"No drive: Y_S,A {static_yield_A:.6f}, Y_S,B {static_yield_B:.6f}, gap D {abs(static_yield_A - static_yield_B):.6f}, "
          f"pairs needed {pairs_needed(static_bhattacharyya, ERROR_PROBABILITY):.0f}")
    print(f"  seed  rounds  kept  final cost   Y_S,A     gap D     pairs needed (error <= {ERROR_PROBABILITY})  "
          f"frequencies (MHz)")
    for random_seed in RANDOM_SEEDS:
        outcome = outcomes[random_seed]
        driven_bhattacharyya = bhattacharyya_of((outcome["driven_yield_A"], outcome["driven_triplet_yield_A"]),
                                                (outcome["driven_yield_B"], outcome["driven_triplet_yield_B"]))
        print(f"  {random_seed:4d}  {len(outcome['accepted']):6d}  {int(np.sum(outcome['accepted'])):4d}"
              f"  {outcome['cost_history'][-1]:10.6f}  {outcome['driven_yield_A']:.6f}"
              f"  {abs(outcome['driven_yield_A'] - outcome['driven_yield_B']):.6f}"
              f"  {pairs_needed(driven_bhattacharyya, ERROR_PROBABILITY):12.0f}                "
              f"{np.round(outcome['frequencies_MHz'], 3).tolist()}")
    return dict(static_yield_A=static_yield_A, static_triplet_yield_A=static_triplet_yield_A,
                static_yield_B=static_yield_B, static_triplet_yield_B=static_triplet_yield_B)


# new
def print_contrast_summary(outcomes):
    static_contrast_A, static_contrast_B = both_contrasts(None)
    print(f"No drive: contrast C_A {static_contrast_A:.6f}, C_B {static_contrast_B:.6f}, "
          f"|C_A - C_B| {abs(static_contrast_A - static_contrast_B):.6f}")
    print("  seed  rounds  kept  final cost   C_A       C_B       |C_A - C_B|  frequencies (MHz)")
    for random_seed in RANDOM_SEEDS:
        outcome = outcomes[random_seed]
        print(f"  {random_seed:4d}  {len(outcome['accepted']):6d}  {int(np.sum(outcome['accepted'])):4d}"
              f"  {outcome['cost_history'][-1]:10.6f}  {outcome['driven_contrast_A']:.6f}"
              f"  {outcome['driven_contrast_B']:.6f}"
              f"  {abs(outcome['driven_contrast_A'] - outcome['driven_contrast_B']):.6f}     "
              f"{np.round(outcome['frequencies_MHz'], 3).tolist()}")
    return dict(static_contrast_A=static_contrast_A, static_contrast_B=static_contrast_B)


# %% run
# new
if __name__ == "__main__":
    if GOAL not in ("disruption", "discrimination") \
            or DISCRIMINATION_MEASURE not in ("gap", "bhattacharyya", "contrast") \
            or CAP_METHOD not in ("peak", "power") or STARTING_FREQUENCIES_FROM not in ("action spectrum", "random"):
        raise ValueError("Check GOAL, DISCRIMINATION_MEASURE, CAP_METHOD and STARTING_FREQUENCIES_FROM")
    if (SEARCH_LEBEDEV_ORDER == 0) != (FINAL_LEBEDEV_ORDER == 0):
        raise ValueError("SEARCH_LEBEDEV_ORDER and FINAL_LEBEDEV_ORDER must both be 0 (no averaging) or both not 0")
    if DISCRIMINATION_MEASURE == "contrast" and (GOAL != "discrimination" or SEARCH_LEBEDEV_ORDER != 0):
        raise ValueError("The contrast measure needs GOAL = 'discrimination' and no orientation averaging")

    # an interrupted run with exactly these settings picks up from its checkpoints
    settings = dict(
        goal=GOAL, discrimination_measure=DISCRIMINATION_MEASURE, hypothesis_A=repr(HYPOTHESIS_A),
        hypothesis_B=repr(HYPOTHESIS_B), amplitude_cap_MHz=AMPLITUDE_CAP_MHz, cap_method=CAP_METHOD,
        peak_check_times_us=PEAK_CHECK_TIMES_US, times_us=TIMES_US, max_step_us=MAX_STEP_US,
        number_of_frequencies=NUMBER_OF_FREQUENCIES, starting_frequencies_from=STARTING_FREQUENCIES_FROM,
        random_start_max_frequency_MHz=RANDOM_START_MAX_FREQUENCY_MHz, seed_bin_width_MHz=SEED_BIN_WIDTH_MHz,
        seed_max_frequency_MHz=SEED_MAX_FREQUENCY_MHz, seed_min_separation_MHz=SEED_MIN_SEPARATION_MHz,
        super_iterations=SUPER_ITERATIONS, frequency_perturbation_MHz=FREQUENCY_PERTURBATION_MHz,
        max_nelder_mead_iterations=MAX_NELDER_MEAD_ITERATIONS, random_seeds=RANDOM_SEEDS,
        patience_super_iterations=PATIENCE_SUPER_ITERATIONS, search_lebedev_order=SEARCH_LEBEDEV_ORDER,
        final_lebedev_order=FINAL_LEBEDEV_ORDER, contrast_polar_angles_rad=CONTRAST_FIELD_DIRECTIONS[0],
        contrast_azimuths_rad=CONTRAST_FIELD_DIRECTIONS[1])
    results_folder = REPO_FOLDER / "results"
    run_id, run_checkpoint_folder = start_or_resume_run(results_folder, GOAL, settings)

    run_start = time.perf_counter()
    outcomes = {}
    if NUMBER_OF_PROCESSES == 1:
        for random_seed in RANDOM_SEEDS:
            outcomes[random_seed] = run_one_seed(random_seed, run_checkpoint_folder)
    else:
        with ProcessPoolExecutor(max_workers=NUMBER_OF_PROCESSES) as pool:
            seed_of = {pool.submit(run_one_seed, random_seed, run_checkpoint_folder): random_seed
                       for random_seed in RANDOM_SEEDS}
            for search in as_completed(seed_of):
                outcomes[seed_of[search]] = search.result()
                print(f"Random seed {seed_of[search]} finished ({len(outcomes)} of {len(RANDOM_SEEDS)}), "
                      f"{format_duration(time.perf_counter() - run_start)} elapsed", flush=True)
    print(f"All searches finished in {format_duration(time.perf_counter() - run_start)}")
    if FINAL_ORIENTATIONS is None:
        print("Molecules as given (no orientation averaging)")
    else:
        print(f"Averaged over orientations: {len(SEARCH_ORIENTATIONS[2])} directions while searching (the final cost "
              f"column), {len(FINAL_ORIENTATIONS[2])} for every yield, gap and pairs needed below")

    if DISCRIMINATION_MEASURE == "contrast":
        static_outcome = print_contrast_summary(outcomes)
    else:
        static_outcome = print_yield_summary(outcomes)
    best_random_seed = min(RANDOM_SEEDS, key=lambda random_seed: outcomes[random_seed]["final_cost"])
    best = outcomes[best_random_seed]
    print(f"Best: random seed {best_random_seed}")
    for frequency_MHz, amplitude_MHz, phase_rad in zip(best["frequencies_MHz"], best["amplitudes_MHz"],
                                                        best["phases_rad"]):
        print(f"  {frequency_MHz:8.3f} MHz   amplitude {amplitude_MHz:8.4f} MHz   phase {phase_rad:7.4f} rad")

    # best seed under the plain names, and every seed as random_seed_<n>_<name>
    FORMAT_VERSION = 7
    outcome = dict(**static_outcome, best_random_seed=best_random_seed, **best,
                   **{f"random_seed_{random_seed}_{name}": value for random_seed in RANDOM_SEEDS
                      for name, value in outcomes[random_seed].items()})
    save_run(results_folder, run_id, FORMAT_VERSION, settings, outcome)
    save_quick_look(results_folder, run_id, "cost", "super-iteration (0 = start)", "best cost so far (lower is better)",
                    [(np.arange(len(outcomes[random_seed]["cost_history"])), outcomes[random_seed]["cost_history"],
                      f"random seed {random_seed}") for random_seed in RANDOM_SEEDS])
    finish_run(results_folder, run_id)
    print(f"Saved results/{run_id}.npz and results/{run_id}_cost.png")
