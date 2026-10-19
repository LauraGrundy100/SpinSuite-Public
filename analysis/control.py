"""RF drives: describing, capping, seeding, scoring and optimising a drive.

A drive is a sum of sines, Σ_i a_i · sin(2π f_i t + φ_i), acting on S1x + S2x (both electrons, along x).
Units: frequencies and amplitudes in MHz, phases in rad, times in µs. Amplitudes become rad/µs in one
place only, drive_rad_per_us(), which is the boundary into engine.py.

Two ways to cap the strength, both from the old repo:
    L-infinity (peak)  scale so the waveform's highest point equals the cap (DISCO, APEX).
    L2 (power)         scale so √(Σ a_i²) equals the cap, which fixes the average power Σ a_i²/2 (FORGE).

Starting frequencies come from the action spectrum (after Hiscock et al. 2017): the transitions of the
static Hamiltonian that the drive can cause and that change how much singlet the pair has.

Scores (costs) are numbers to minimise: the singlet yield for disruption, and for discrimination minus the yield
gap, minus √B, the square root of the Bhattacharyya distance (which allows for counting noise), or minus the
difference in contrast (how much each pair's recombination yield changes with field direction).

The optimiser is dCRAB-inspired (not canonical dCRAB): a greedy loop of super-iterations, each nudging all
frequencies at random, then tuning amplitudes and phases by Nelder-Mead, keeping the result only if it is better.
It can save its state (a checkpoint) after every super-iteration and carry on from it after an interruption.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize

from engine import bhattacharyya_distance, outcome_probabilities, recombination_contrast, yields, yields_together
from model import angular_frequency, drive_operator, oriented_hypothesis, recombination_projector, static_hamiltonian
from results import progress_line, read_npz, save_checkpoint


@dataclass(frozen=True)
class Drive:
    """An RF drive Σ_i amplitudes_MHz[i] · sin(2π frequencies_MHz[i] t + phases_rad[i]), already capped.

    Amplitudes are physical (after normalisation). A negative amplitude is the same as adding π to the phase.
    """
    frequencies_MHz: np.ndarray
    amplitudes_MHz: np.ndarray
    phases_rad: np.ndarray


def waveform(times_us, frequencies_MHz, amplitudes, phases_rad):
    """Σ_i amplitudes[i] · sin(2π frequencies_MHz[i] t + phases_rad[i]) at times in µs, in the units of `amplitudes`."""
    times_us = np.asarray(times_us, dtype=float)
    total = np.zeros_like(times_us)
    for frequency_MHz, amplitude, phase_rad in zip(frequencies_MHz, amplitudes, phases_rad):
        total += amplitude * np.sin(angular_frequency(frequency_MHz) * times_us + phase_rad)
    return total


def linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz, peak_check_times_us):
    """Drive whose largest |value| over peak_check_times_us (µs) equals amplitude_cap_MHz (L-infinity normalisation).

    raw_amplitudes set only the ratios between components; their overall size is removed. The peak is
    found by sampling, so between samples the drive can rise slightly above the cap.
    The old repo checked the peak on 0–12 µs with 20000 points.
    """
    raw_amplitudes = np.asarray(raw_amplitudes, dtype=float)
    peak = np.max(np.abs(waveform(peak_check_times_us, frequencies_MHz, raw_amplitudes, phases_rad)))
    if peak < 1e-12:  # nothing to scale. The old code used scale 1 here, which also gives a negligible drive
        scale = 0.0
    else:
        scale = amplitude_cap_MHz / peak
    return Drive(np.array(frequencies_MHz, dtype=float), scale * raw_amplitudes, np.array(phases_rad, dtype=float))


def l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz):
    """Drive whose amplitudes satisfy √(Σ a_i²) = amplitude_cap_MHz (L2 normalisation, from FORGE).

    For distinct frequencies the average power is Σ a_i²/2, so this fixes the power: N equal components
    each get cap/√N. A single sine gets amplitude = cap, as under L-infinity.
    """
    raw_amplitudes = np.asarray(raw_amplitudes, dtype=float)
    size = np.sqrt(np.sum(raw_amplitudes ** 2))
    if size < 1e-12:  # nothing to scale, as in linf_drive
        scale = 0.0
    else:
        scale = amplitude_cap_MHz / size
    return Drive(np.array(frequencies_MHz, dtype=float), scale * raw_amplitudes, np.array(phases_rad, dtype=float))


def drive_rad_per_us(drive):
    """The function f(t) that engine.yields takes: the drive in rad/µs at an array of times in µs.

    This is the only place drive amplitudes are converted from MHz to rad/µs.
    """
    amplitudes_rad_per_us = angular_frequency(drive.amplitudes_MHz)

    def drive_function(times_us):
        return waveform(times_us, drive.frequencies_MHz, amplitudes_rad_per_us, drive.phases_rad)

    return drive_function


# %% Starting frequencies: action-spectrum seeding

def action_spectrum(hypothesis, bin_width_MHz, max_frequency_MHz, orientations=None):
    """Score each frequency bin (MHz) by how strongly a drive there should change the singlet population.

    For every pair of eigenstates i, j of the static Hamiltonian H (Hermitian, no recombination term, as in
    the old code), the score |⟨i|X|j⟩|² × |s_i − s_j| goes into the bin holding the transition frequency
    |E_i − E_j|/2π. X = S1x + S2x is the drive operator and s_i = ⟨i|P_S|i⟩ is state i's singlet character.
    Returns bin centres (MHz) and scores scaled so the largest is 1. Ported from All DISCO/DISCO.py lines 280-308.
    orientations: None for the molecule as given; or (polar angles, azimuths, weights) from engine.orientation_grid,
    to add up every direction's scores with its weight (before the scaling), for a sample of all orientations.
    """
    if orientations is None:
        weighted_hypotheses = [(hypothesis, 1.0)]
    else:
        polar_angles_rad, azimuths_rad, weights = orientations
        weighted_hypotheses = [(oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad), weight)
                               for polar_angle_rad, azimuth_rad, weight in zip(polar_angles_rad, azimuths_rad, weights)]

    number_of_bins = int(max_frequency_MHz / bin_width_MHz)
    bin_edges_MHz = np.linspace(0, max_frequency_MHz, number_of_bins + 1)
    bin_centres_MHz = 0.5 * (bin_edges_MHz[:-1] + bin_edges_MHz[1:])
    scores = np.zeros(number_of_bins)
    for direction_hypothesis, weight in weighted_hypotheses:
        energies_rad_per_us, eigenvectors = np.linalg.eigh(static_hamiltonian(direction_hypothesis))
        drive_elements = eigenvectors.conj().T @ drive_operator(direction_hypothesis) @ eigenvectors  # ⟨i|X|j⟩
        singlet_character = np.real(np.diag(  # character in the recombination state: the singlet unless set otherwise
            eigenvectors.conj().T @ recombination_projector(direction_hypothesis) @ eigenvectors))
        for i in range(len(energies_rad_per_us)):
            for j in range(i + 1, len(energies_rad_per_us)):
                # The one conversion back from rad/µs to MHz in the repo: a transition frequency, for binning.
                frequency_MHz = abs(energies_rad_per_us[i] - energies_rad_per_us[j]) / (2 * np.pi)
                bin_index = int(frequency_MHz / bin_width_MHz)
                if bin_index < number_of_bins:
                    scores[bin_index] += weight * (abs(drive_elements[i, j]) ** 2
                                                   * abs(singlet_character[i] - singlet_character[j]))
    if scores.max() > 0:
        scores = scores / scores.max()
    return bin_centres_MHz, scores


def action_spectrum_seeds(bin_centres_MHz, scores, number_of_seeds, min_separation_MHz):
    """Starting frequencies (MHz): bin centres in order of decreasing score, skipping any closer than
    min_separation_MHz to one already chosen. Ported from All DISCO/DISCO.py lines 311-323.

    Ties are broken by np.argsort, exactly as in the old code. If too few bins are far enough apart this
    raises an error; the old code padded with 5, 10, ... MHz instead, which never happened with 40 bins.
    """
    seeds_MHz = []
    for index in np.argsort(scores)[::-1]:
        frequency_MHz = bin_centres_MHz[index]
        if all(abs(frequency_MHz - seed) >= min_separation_MHz for seed in seeds_MHz):
            seeds_MHz.append(frequency_MHz)
        if len(seeds_MHz) == number_of_seeds:
            return np.array(seeds_MHz)
    raise ValueError(f"Only {len(seeds_MHz)} bins are far enough apart; asked for {number_of_seeds} seeds")


def random_starting_frequencies(random_seed, number_of_frequencies, max_frequency_MHz):
    """Starting frequencies (MHz) drawn uniformly from 0 to max_frequency_MHz, ignoring the physics.

    For testing whether action-spectrum seeding helps. Uses NumPy's newer generator (default_rng), a different
    kind from the optimiser's nudge generator, so these are unrelated to the nudges even with the same seed.
    """
    return np.random.default_rng(random_seed).uniform(0.0, max_frequency_MHz, size=number_of_frequencies)


def random_drive_parameters(random_numbers, number_of_frequencies, max_frequency_MHz):
    """One random, unoptimised drive for a null test: frequencies uniform in 0 to max_frequency_MHz (MHz), raw
    amplitudes uniform in 0 to 1, phases uniform in 0 to 2π (rad). Apply the cap (linf_drive or l2_drive) after.

    random_numbers is a np.random.default_rng generator; drawing every drive from one generator in turn makes
    the whole set reproducible from its seed.
    """
    frequencies_MHz = random_numbers.uniform(0.0, max_frequency_MHz, size=number_of_frequencies)
    raw_amplitudes = random_numbers.uniform(0.0, 1.0, size=number_of_frequencies)
    phases_rad = random_numbers.uniform(0.0, 2 * np.pi, size=number_of_frequencies)
    return frequencies_MHz, raw_amplitudes, phases_rad


# %% Scoring a drive (costs to minimise)

def disruption_cost(hypothesis, drive, times_us, max_step_us, orientations=None):
    """Cost for disruption: the singlet yield Y_S of one hypothesis under the drive (dimensionless). Lower is better.

    orientations (here and in the two costs below): None for the molecule as given, or a grid from
    engine.orientation_grid to score the orientation-averaged yields of a sample instead.
    """
    singlet_yield, _ = yields(hypothesis, drive_rad_per_us(drive), times_us, max_step_us, orientations)
    return singlet_yield


def discrimination_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations=None):
    """Cost for discrimination: −|Y_S,A − Y_S,B|, both hypotheses under the same drive (dimensionless).

    Minimising it maximises the gap D = |Y_S,A − Y_S,B|, so D = −cost. Both pairs are simulated in one pass.
    """
    (singlet_yield_A, _), (singlet_yield_B, _) = yields_together([hypothesis_A, hypothesis_B],
                                                                 drive_rad_per_us(drive), times_us, max_step_us,
                                                                 orientations)
    return -abs(singlet_yield_A - singlet_yield_B)


def bhattacharyya_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations=None):
    """Cost for discrimination under counting noise: −√B, B the Bhattacharyya distance per pair (dimensionless).

    B compares all three outcomes (singlet product, triplet product, not yet recombined) of both hypotheses under
    the same drive, simulated in one pass. The square root keeps the cost on a scale like the yield gap (B grows
    like the gap squared), so Nelder-Mead's fixed stopping tolerance stays sensible. It does not change the best drive.
    """
    (singlet_yield_A, triplet_yield_A), (singlet_yield_B, triplet_yield_B) = yields_together(
        [hypothesis_A, hypothesis_B], drive_rad_per_us(drive), times_us, max_step_us, orientations)
    return -np.sqrt(bhattacharyya_distance(outcome_probabilities(singlet_yield_A, triplet_yield_A),
                                           outcome_probabilities(singlet_yield_B, triplet_yield_B)))


def contrast_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, field_directions):
    """Cost for discrimination by contrast: −|C_A − C_B|, C the contrast (highest minus lowest recombination yield
    over field_directions, engine.recombination_contrast), both hypotheses under the same drive (dimensionless).

    field_directions: (polar angles, azimuths) in rad, in the molecule's frame. Large when the drive removes the
    compass effect of one pair and leaves (or enlarges) the other's.
    """
    contrast_A, contrast_B = (recombination_contrast(hypothesis, drive_rad_per_us(drive), times_us, max_step_us,
                                                     field_directions) for hypothesis in (hypothesis_A, hypothesis_B))
    return -abs(contrast_A - contrast_B)


# %% Optimiser (dCRAB-inspired greedy loop)

@dataclass(frozen=True)
class OptimisationResult:
    """Outcome of optimise_drive: the best frequencies (MHz), raw amplitudes and phases (rad) found.

    cost_history[0] is the starting cost and cost_history[k] the best cost after super-iteration k.
    accepted[k - 1] says whether super-iteration k improved; evaluations[k - 1] counts its cost calls.
    nelder_mead_costs: the cost of every call, in order, all super-iterations one after another (the first
    evaluations[0] belong to super-iteration 1, and so on); NaN where not recorded (checkpoints from before 2026-10-14).
    If the search stopped early (patience), these are shorter than the number of super-iterations asked for.
    """
    frequencies_MHz: np.ndarray
    raw_amplitudes: np.ndarray
    phases_rad: np.ndarray
    cost_history: np.ndarray
    accepted: np.ndarray
    evaluations: np.ndarray
    nelder_mead_costs: np.ndarray


@dataclass
class SearchState:
    """Everything the search needs to carry on where it left off: what a checkpoint holds.

    The best drive so far (frequencies in MHz, raw amplitudes, phases in rad), the histories that become an
    OptimisationResult, the patience count, the private random generator, and the time spent (s).
    Not frozen: one_super_iteration updates it in place.
    """
    frequencies_MHz: np.ndarray
    raw_amplitudes: np.ndarray
    phases_rad: np.ndarray
    cost_history: list
    accepted: list
    evaluations: list
    nelder_mead_costs: list
    super_iterations_without_improvement: int
    random_numbers: np.random.RandomState
    elapsed_seconds: float


def new_search_state(cost, seed_frequencies_MHz, random_seed):
    """Search state at the start: the seed frequencies (MHz), raw amplitudes 1, phases 0, and their cost."""
    number_of_terms = len(seed_frequencies_MHz)
    frequencies_MHz = np.array(seed_frequencies_MHz, dtype=float)
    raw_amplitudes, phases_rad = np.ones(number_of_terms), np.zeros(number_of_terms)
    start_cost = cost(frequencies_MHz, raw_amplitudes, phases_rad)
    random_numbers = np.random.RandomState(random_seed)  # the old np.random.seed generator, private to this run
    return SearchState(frequencies_MHz, raw_amplitudes, phases_rad, [start_cost], [], [], [], 0, random_numbers, 0.0)


def one_super_iteration(state, cost, frequency_perturbation_MHz, max_nelder_mead_iterations):
    """One round of the search, updating state in place. Returns the candidate's cost (kept or not).

    Nudges every frequency by a uniform random amount in ±frequency_perturbation_MHz (then takes abs), tunes
    amplitudes and phases by Nelder-Mead from the current ones, and keeps the result only if its cost is lower.
    The cost of every Nelder-Mead call is added to state.nelder_mead_costs (a record only; it changes nothing).
    """
    number_of_terms = len(state.frequencies_MHz)
    nudge_MHz = state.random_numbers.uniform(-frequency_perturbation_MHz, frequency_perturbation_MHz,
                                             size=number_of_terms)
    candidate_frequencies_MHz = np.abs(state.frequencies_MHz + nudge_MHz)

    def cost_at_candidate(amplitudes_then_phases):
        value = cost(candidate_frequencies_MHz, amplitudes_then_phases[:number_of_terms],
                     amplitudes_then_phases[number_of_terms:])
        state.nelder_mead_costs.append(value)
        return value

    result = scipy.optimize.minimize(  # tolerances are the old repo's values
        cost_at_candidate, np.concatenate([state.raw_amplitudes, state.phases_rad]), method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": max_nelder_mead_iterations})
    improved = result.fun < state.cost_history[-1]
    if improved:
        state.frequencies_MHz = candidate_frequencies_MHz
        state.raw_amplitudes, state.phases_rad = result.x[:number_of_terms], result.x[number_of_terms:]
    state.cost_history.append(result.fun if improved else state.cost_history[-1])
    state.accepted.append(improved)
    state.evaluations.append(result.nfev)
    state.super_iterations_without_improvement = 0 if improved else state.super_iterations_without_improvement + 1
    return result.fun


def save_search_state(path, state):
    """Write a search state to a checkpoint file (.npz). The generator is saved as its internal state
    (624 numbers and a position: its bookmark), so it can be put back exactly."""
    _, generator_keys, generator_position, generator_has_gauss, generator_cached_gaussian = \
        state.random_numbers.get_state()
    save_checkpoint(path, checkpoint_format_version=2, frequencies_MHz=state.frequencies_MHz,
                    raw_amplitudes=state.raw_amplitudes, phases_rad=state.phases_rad,
                    cost_history=state.cost_history, accepted=state.accepted, evaluations=state.evaluations,
                    nelder_mead_costs=np.array(state.nelder_mead_costs, dtype=float),
                    super_iterations_without_improvement=state.super_iterations_without_improvement,
                    elapsed_seconds=state.elapsed_seconds, generator_keys=generator_keys,
                    generator_position=generator_position, generator_has_gauss=generator_has_gauss,
                    generator_cached_gaussian=generator_cached_gaussian)


def load_search_state(path):
    """Read a search state written by save_search_state, with its random generator restored exactly.
    Version 1 checkpoints (before 2026-10-14) did not record nelder_mead_costs: those calls are filled with NaN."""
    saved = read_npz(path)
    if "nelder_mead_costs" not in saved:
        saved["nelder_mead_costs"] = np.full(int(np.sum(saved["evaluations"])), np.nan)
    random_numbers = np.random.RandomState()
    random_numbers.set_state(("MT19937", saved["generator_keys"], saved["generator_position"],
                              saved["generator_has_gauss"], saved["generator_cached_gaussian"]))
    return SearchState(saved["frequencies_MHz"], saved["raw_amplitudes"], saved["phases_rad"],
                       list(saved["cost_history"]), list(saved["accepted"]), list(saved["evaluations"]),
                       list(saved["nelder_mead_costs"]), saved["super_iterations_without_improvement"],
                       random_numbers, saved["elapsed_seconds"])


def optimise_drive(cost, seed_frequencies_MHz, super_iterations, frequency_perturbation_MHz,
                   max_nelder_mead_iterations, random_seed, patience_super_iterations, checkpoint_path=None):
    """Minimise cost(frequencies_MHz, raw_amplitudes, phases_rad) by the dCRAB-inspired greedy loop.

    Starts at the seed frequencies with raw amplitudes 1 and phases 0, then repeats one_super_iteration.
    Ported from All DISCO/DISCO.py lines 353-398. Same random generator and seed as the old code, so its
    nudges are reproduced exactly. Stops early after patience_super_iterations in a row without improvement
    (as StatDISCO_v2 did); setting it to super_iterations or more turns this off, as in the old runs.
    If checkpoint_path is given, the state is saved there after every super-iteration, and if that file
    already exists the search carries on from it, giving exactly the result of an uninterrupted search.
    Printed lines then start with the checkpoint's name, e.g. "[random_seed_3]", to tell side-by-side searches apart.
    """
    label = "" if checkpoint_path is None else f"[{Path(checkpoint_path).stem}] "
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = load_search_state(checkpoint_path)
        print(f"  {label}resuming: {len(state.accepted)} super-iterations already done")
    else:
        state = new_search_state(cost, seed_frequencies_MHz, random_seed)
        print(f"  {label}start cost = {state.cost_history[0]:.6f}", flush=True)
    start_time = time.perf_counter() - state.elapsed_seconds  # so time before an interruption counts too

    while (len(state.accepted) < super_iterations
           and state.super_iterations_without_improvement < patience_super_iterations):
        candidate_cost = one_super_iteration(state, cost, frequency_perturbation_MHz, max_nelder_mead_iterations)
        state.elapsed_seconds = time.perf_counter() - start_time
        if checkpoint_path is not None:
            save_search_state(checkpoint_path, state)
        print(f"  {label}{progress_line('super-iteration', len(state.accepted), super_iterations, state.elapsed_seconds)}"
              f": candidate cost = {candidate_cost:.6f} ({'kept' if state.accepted[-1] else 'not kept'})", flush=True)
    if state.super_iterations_without_improvement >= patience_super_iterations:
        print(f"  {label}stopped: {patience_super_iterations} super-iterations in a row without improvement")

    return OptimisationResult(state.frequencies_MHz, state.raw_amplitudes, state.phases_rad,
                              np.array(state.cost_history), np.array(state.accepted), np.array(state.evaluations),
                              np.array(state.nelder_mead_costs, dtype=float))
