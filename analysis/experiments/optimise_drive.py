"""Optimise an RF drive for disruption (lower one pair's singlet yield) or discrimination (make two pairs easier to
tell apart, by the yield gap or by the Bhattacharyya distance under counting noise).

One search per random seed, side by side in separate processes. Starting frequencies come from the action spectrum
(or, to test whether that helps, at random), the search is the dCRAB-inspired loop in control.py, and all searches
are saved together to results/ with the settings. Each search saves a checkpoint after every super-iteration;
starting the script again with the same settings carries on where it stopped.
Side-by-side searches must be started from the terminal (python experiments/optimise_drive.py). In the VS Code
interactive window set NUMBER_OF_PROCESSES = 1, run the cells above the last one, then the last cell.
"""

import os

# One thread per process, so side-by-side searches do not compete for cores. Must come before importing NumPy.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Make model.py, engine.py and control.py importable, whether run as a script or cell by cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_FOLDER = HERE if (HERE / "model.py").exists() else HERE.parent
sys.path.insert(0, str(REPO_FOLDER))

from control import (action_spectrum, action_spectrum_seeds, bhattacharyya_cost, contrast_cost, discrimination_cost,
                     disruption_cost, drive_rad_per_us, l2_drive, linf_drive, optimise_drive,
                     random_starting_frequencies)
from engine import (bhattacharyya_distance, orientation_grid, outcome_probabilities, pairs_needed,
                    recombination_contrast, yields_together)
from model import baseline_hypothesis, ciss_density_matrix, field_direction_hypothesis, flavin_tryptophan_hypothesis
from results import finish_run, format_duration, save_quick_look, save_run, start_or_resume_run

# %% Settings

GOAL = "discrimination"  # "disruption": minimise Y_S of pair A. "discrimination": make pairs A and B easier to tell apart
DISCRIMINATION_MEASURE = "contrast"  # "gap": maximise |Y_S,A − Y_S,B| (as the old runs). "bhattacharyya": maximise √B.
#                                     "contrast": maximise |C_A − C_B|, C = highest minus lowest recombination yield
#                                     over CONTRAST_FIELD_DIRECTIONS (Daniel's measure; needs SEARCH_LEBEDEV_ORDER = 0)
ERROR_PROBABILITY = 0.01  # for the printed "pairs needed" (gap and bhattacharyya only)
# Daniel's flavin-tryptophan pairs (from 2026-10-15; assumptions listed in flavin_tryptophan_hypothesis, model.py).
# Earlier runs used DISCO's placeholder pairs (may be stale): to 2026-10-08 baseline_hypothesis(0.0) against
# baseline_hypothesis(0.3); then baseline_hypothesis(0.3) against baseline_hypothesis(0.3).variant(
# name="baseline, delta=0.3, CISS chi=0.1pi", electron_initial_state=ciss_density_matrix(0.1 * np.pi)), cap 7.0 MHz
# (0.25 × A_iso), TIMES_US = np.linspace(0, 6, 2000), lebedev orders 11 and 23, measure "bhattacharyya".
FIELD_STRENGTH_uT = 50.0  # geomagnetic field strength (µT). Assumption: Daniel gave no strength
HYPOTHESIS_A = flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, FIELD_STRENGTH_uT)  # classical: chi 0, k_r = k_f = 1 per µs
HYPOTHESIS_B = flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, FIELD_STRENGTH_uT)  # CISS: chi pi/2, k_r 10, k_f 1
# Field directions for the contrast, (polar angles, azimuths) in rad in the molecule's frame: the lowest and highest
# directions of both pairs from the static scan field_scan_2026-10-15_01 (rounded to 1e-4 rad). The two pairs'
# extremes lie far apart, so all four are used: each pair's no-drive contrast is then its full contrast.
CONTRAST_FIELD_DIRECTIONS = (np.array([1.4393, 0.4242, 2.6949, 1.3668]),  # pair A lowest, A highest,
                             np.array([-2.6991, 3.0531, -0.5672, 3.127]))  # B lowest, B highest

AMPLITUDE_CAP_MHz = 0.25 * HYPOTHESIS_A.zeeman_frequency_MHz  # placeholder cap: 0.25 × Zeeman (about 0.35 MHz). The
#                                                              old 7 MHz was 0.25 × the old 28 MHz hyperfine
CAP_METHOD = "peak"  # "peak" (L-infinity: highest point = cap) or "power" (L2: √(Σ a²) = cap)
PEAK_CHECK_TIMES_US = np.linspace(0, 12, 20000)  # where the peak cap looks; this is the old repo's grid

TIMES_US = np.linspace(0, 12, 4001)  # yield window and output grid (µs): about 12 lifetimes of 1/k_f (Daniel: 10 to
#                                      12), 3 ns grid. Old runs: np.linspace(0, 6, 2000), DISCO's grid
MAX_STEP_US = 0.00151  # largest simulation step (µs). On this grid: 2 steps of 1.50 ns per output interval.
#                       Old system: yield error at most 1e-5 for drives up to 100 MHz (scratch, 2026-10-08); the
#                       flavin-tryptophan pair agrees with the reference to about 1e-7 (tests/test_engine.py)

# Orientations of the molecule. 0 = no averaging: the molecules as given, axis along the field (every run before
# 2026-10-13). Otherwise a Lebedev order (engine.orientation_grid): 11 = 50 directions, 17 = 110, 23 = 194.
# Must be 0 for the flavin-tryptophan pairs, which have no symmetry axis (IDEAS.md idea 21).
SEARCH_LEBEDEV_ORDER = 0  # directions while searching (exploratory run 2026-10-14: 11, 50 directions)
FINAL_LEBEDEV_ORDER = 0  # directions for the reported yields of no drive and of each final drive (2026-10-14: 23, 194).
#                          Scratch 2026-10-13, 15 drives: 50 directions usually within 6% of 194 in pairs needed, 110
#                          within 3% (one unoptimised drive: 27% and 13%). 0 if SEARCH_LEBEDEV_ORDER is 0

NUMBER_OF_FREQUENCIES = 4  # number of frequencies in the drive
STARTING_FREQUENCIES_FROM = "action spectrum"  # "action spectrum" (the physics), or "random": uniform in 0 to
#                                                RANDOM_START_MAX_FREQUENCY_MHz, different for each random seed
RANDOM_START_MAX_FREQUENCY_MHz = 40.0  # used only for random starts
SEED_BIN_WIDTH_MHz = 1.0  # action-spectrum bin width
SEED_MAX_FREQUENCY_MHz = 60.0  # highest action-spectrum bin edge. 60 for the flavin-tryptophan pairs: 99% of their
#                                action-spectrum score lies below 60 MHz, only 62% below 40 (scratch, 2026-10-15).
#                                40 for every run before 2026-10-15
SEED_MIN_SEPARATION_MHz = 2.0  # action-spectrum starting frequencies closer than this are skipped. Every run before
#                                2026-10-13 used 0.5: set 0.5 to compare like with like with them (IDEAS.md idea 2)

SUPER_ITERATIONS = 5  # most rounds of the search loop (old DISCO runs: 20). 5 for the first flavin-tryptophan run:
#                       one contrast evaluation takes about 3.7 s (scratch 2026-10-15), so a round takes 0.5-1.5 h
PATIENCE_SUPER_ITERATIONS = 5  # stop after this many rounds in a row without improvement (= SUPER_ITERATIONS: never)
FREQUENCY_PERTURBATION_MHz = 5.0  # each round nudges every frequency by up to ± this
MAX_NELDER_MEAD_ITERATIONS = 1000  # cap on fine-tuning steps per round (old value 1000; a lower cap lost ~6% of D
#                                    in a round that reached it, scratch measurement 2026-10-08)
RANDOM_SEEDS = [1, 2, 3, 4, 5]  # one search per random seed, which fixes its nudges (and its start, if random).
#                                    [123] reproduces the old runs' nudges
NUMBER_OF_PROCESSES = 5  # searches side by side (this laptop: 6 cores, 12 threads; one core left free); 1 = one after another.
#                          Results do not depend on it, so it is not saved and changing it does not stop a resume

########################################################################################################
########################################################################################################

# %% Starting frequencies, cost function, and one search (one random seed)

def orientations(lebedev_order):
    """The orientation grid for a Lebedev order, or None (no averaging, molecules as given) for 0."""
    return None if lebedev_order == 0 else orientation_grid(lebedev_order)


SEARCH_ORIENTATIONS = orientations(SEARCH_LEBEDEV_ORDER)
FINAL_ORIENTATIONS = orientations(FINAL_LEBEDEV_ORDER)


def action_spectrum_starting_frequencies():
    """Starting frequencies (MHz) from the action spectrum, over the search's orientations: of pair A for
    disruption, or for discrimination of whichever pair scores higher in each bin (element-wise maximum, as DISCO did).
    For the contrast, the maximum is also taken over the contrast's field directions.
    """
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


def make_drive(frequencies_MHz, raw_amplitudes, phases_rad):
    """Apply the strength cap chosen in the settings."""
    if CAP_METHOD == "peak":
        return linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz, PEAK_CHECK_TIMES_US)
    return l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz)


def cost(frequencies_MHz, raw_amplitudes, phases_rad):
    """The number the search minimises: Y_S of pair A (disruption), or for discrimination −|Y_S,A − Y_S,B| (gap),
    −√B (bhattacharyya), over the search's orientations, or −|C_A − C_B| (contrast)."""
    drive = make_drive(frequencies_MHz, raw_amplitudes, phases_rad)
    if GOAL == "disruption":
        return disruption_cost(HYPOTHESIS_A, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)
    if DISCRIMINATION_MEASURE == "gap":
        return discrimination_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)
    if DISCRIMINATION_MEASURE == "contrast":
        return contrast_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, CONTRAST_FIELD_DIRECTIONS)
    return bhattacharyya_cost(HYPOTHESIS_A, HYPOTHESIS_B, drive, TIMES_US, MAX_STEP_US, SEARCH_ORIENTATIONS)


def both_yields(drive):
    """(Y_S, Y_T) of pair A and of pair B under a drive (None: no drive), in one pass, over the final orientations."""
    return yields_together([HYPOTHESIS_A, HYPOTHESIS_B], None if drive is None else drive_rad_per_us(drive),
                           TIMES_US, MAX_STEP_US, FINAL_ORIENTATIONS)


def both_contrasts(drive):
    """Contrast C_A and C_B (highest minus lowest recombination yield over CONTRAST_FIELD_DIRECTIONS) under a drive
    (None: no drive)."""
    drive_function = None if drive is None else drive_rad_per_us(drive)
    return tuple(recombination_contrast(hypothesis, drive_function, TIMES_US, MAX_STEP_US, CONTRAST_FIELD_DIRECTIONS)
                 for hypothesis in (HYPOTHESIS_A, HYPOTHESIS_B))


def final_cost(yields_A, yields_B):
    """The search's cost recomputed from both pairs' final yields, (Y_S, Y_T) each, or for the contrast from
    (C_A, C_B); picks the best random seed. Without averaging it equals the search's own last cost."""
    if GOAL == "disruption":
        return yields_A[0]
    if DISCRIMINATION_MEASURE in ("gap", "contrast"):
        return -abs(yields_A[0] - yields_B[0])
    return -np.sqrt(bhattacharyya_of(yields_A, yields_B))


def run_one_seed(random_seed, checkpoint_folder):
    """One search for one random seed (in its own process), checkpointed in checkpoint_folder.
    Returns a dict of results, with the names they are saved under."""
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


def bhattacharyya_of(yields_A, yields_B):
    """B per pair from each pair's (Y_S, Y_T)."""
    return bhattacharyya_distance(outcome_probabilities(*yields_A), outcome_probabilities(*yields_B))


def print_yield_summary(outcomes):
    """Print no drive and one row per random seed (gap and bhattacharyya measures); return the no-drive yields."""
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


def print_contrast_summary(outcomes):
    """Print no drive and one row per random seed (contrast measure); return the no-drive contrasts."""
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


# %% Run every random seed, print a summary, save

if __name__ == "__main__":
    if GOAL not in ("disruption", "discrimination") \
            or DISCRIMINATION_MEASURE not in ("gap", "bhattacharyya", "contrast") \
            or CAP_METHOD not in ("peak", "power") or STARTING_FREQUENCIES_FROM not in ("action spectrum", "random"):
        raise ValueError("Check GOAL, DISCRIMINATION_MEASURE, CAP_METHOD and STARTING_FREQUENCIES_FROM")
    if (SEARCH_LEBEDEV_ORDER == 0) != (FINAL_LEBEDEV_ORDER == 0):
        raise ValueError("SEARCH_LEBEDEV_ORDER and FINAL_LEBEDEV_ORDER must both be 0 (no averaging) or both not 0")
    if DISCRIMINATION_MEASURE == "contrast" and (GOAL != "discrimination" or SEARCH_LEBEDEV_ORDER != 0):
        raise ValueError("The contrast measure needs GOAL = 'discrimination' and no orientation averaging")

    # Every setting is saved with the run. A run with exactly these settings that was interrupted is resumed.
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

    # One search per random seed, side by side
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

    # Summary: no drive, each random seed, and the best one
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

    # Save: the best search under the same names as format version 3, and every search as random_seed_<n>_<name>
    FORMAT_VERSION = 7  # 2: adds patience_super_iterations (version 1 had none, so it ran every round)
    #                    3: adds discrimination_measure (versions 1-2 used the gap) and the triplet yields
    #                    4: one search per random seed (random_seeds replaces random_seed, best_random_seed and
    #                       random_seed_<n>_<name> added), number_of_seeds renamed number_of_frequencies and
    #                       seeds_MHz renamed starting_frequencies_MHz, starting_frequencies_from added
    #                    5: adds search_lebedev_order and final_lebedev_order (versions 1-4 had no averaging and are
    #                       read as 0) and final_cost; the best random seed is the one with the lowest final_cost
    #                    6: adds nelder_mead_costs (the cost of every Nelder-Mead call, rounds one after another;
    #                       split them with evaluations), to study how many calls a round really needs
    #                    7: hypothesis descriptions gain recombination_state and dipolar_tensor_MHz; adds the
    #                       "contrast" measure with contrast_polar_angles_rad and contrast_azimuths_rad (settings),
    #                       static_contrast_A/B and driven_contrast_A/B, which replace the yields for that measure
    outcome = dict(**static_outcome, best_random_seed=best_random_seed, **best,
                   **{f"random_seed_{random_seed}_{name}": value for random_seed in RANDOM_SEEDS
                      for name, value in outcomes[random_seed].items()})
    save_run(results_folder, run_id, FORMAT_VERSION, settings, outcome)
    save_quick_look(results_folder, run_id, "cost", "super-iteration (0 = start)", "best cost so far (lower is better)",
                    [(np.arange(len(outcomes[random_seed]["cost_history"])), outcomes[random_seed]["cost_history"],
                      f"random seed {random_seed}") for random_seed in RANDOM_SEEDS])
    finish_run(results_folder, run_id)  # label its checkpoints finished; keep only the newest three finished runs'
    print(f"Saved results/{run_id}.npz and results/{run_id}_cost.png")
