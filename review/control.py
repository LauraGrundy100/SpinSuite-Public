import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize

from engine import bhattacharyya_distance, outcome_probabilities, recombination_contrast, yields, yields_together
from model import angular_frequency, drive_operator, oriented_hypothesis, recombination_projector, static_hamiltonian
from results import progress_line, read_npz, save_checkpoint

# Drive f(t) = Σ a_i sin(2π f_i t + φ_i) on S1x + S2x: along lab x, with the static field along lab z.
# Frequencies and amplitudes in MHz, phases in rad, times in µs.


# new
@dataclass(frozen=True)
class Drive:
    frequencies_MHz: np.ndarray
    amplitudes_MHz: np.ndarray  # after the cap
    phases_rad: np.ndarray


# old, edited: frequencies in MHz
def waveform(times_us, frequencies_MHz, amplitudes, phases_rad):
    times_us = np.asarray(times_us, dtype=float)
    total = np.zeros_like(times_us)
    for frequency_MHz, amplitude, phase_rad in zip(frequencies_MHz, amplitudes, phases_rad):
        total += amplitude * np.sin(angular_frequency(frequency_MHz) * times_us + phase_rad)
    return total


# old, edited: returns the capped drive; zero peak gives zero drive
def linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz, peak_check_times_us):
    # peak cap: largest |f(t)| on the sampled times equals the cap, so it can overshoot slightly between samples
    raw_amplitudes = np.asarray(raw_amplitudes, dtype=float)
    peak = np.max(np.abs(waveform(peak_check_times_us, frequencies_MHz, raw_amplitudes, phases_rad)))
    if peak < 1e-12:
        scale = 0.0
    else:
        scale = amplitude_cap_MHz / peak
    return Drive(np.array(frequencies_MHz, dtype=float), scale * raw_amplitudes, np.array(phases_rad, dtype=float))


# old, edited: returns the capped drive
def l2_drive(frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz):
    # power cap: √(Σ a²) = cap
    raw_amplitudes = np.asarray(raw_amplitudes, dtype=float)
    size = np.sqrt(np.sum(raw_amplitudes ** 2))
    if size < 1e-12:
        scale = 0.0
    else:
        scale = amplitude_cap_MHz / size
    return Drive(np.array(frequencies_MHz, dtype=float), scale * raw_amplitudes, np.array(phases_rad, dtype=float))


# new
def drive_rad_per_us(drive):
    # the one place drive amplitudes go from MHz to rad/µs
    amplitudes_rad_per_us = angular_frequency(drive.amplitudes_MHz)

    def drive_function(times_us):
        return waveform(times_us, drive.frequencies_MHz, amplitudes_rad_per_us, drive.phases_rad)

    return drive_function


# old, edited: recombination projector, optional orientation average
def action_spectrum(hypothesis, bin_width_MHz, max_frequency_MHz, orientations=None):
    """Bin score Σ |⟨i|X|j⟩|² |p_i - p_j| at |E_i - E_j|/2π, p = ⟨i|P_R|i⟩, eigenstates of H without K; max 1."""
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
        drive_elements = eigenvectors.conj().T @ drive_operator(direction_hypothesis) @ eigenvectors
        singlet_character = np.real(np.diag(
            eigenvectors.conj().T @ recombination_projector(direction_hypothesis) @ eigenvectors))
        for i in range(len(energies_rad_per_us)):
            for j in range(i + 1, len(energies_rad_per_us)):
                frequency_MHz = abs(energies_rad_per_us[i] - energies_rad_per_us[j]) / (2 * np.pi)
                bin_index = int(frequency_MHz / bin_width_MHz)
                if bin_index < number_of_bins:
                    scores[bin_index] += weight * (abs(drive_elements[i, j]) ** 2
                                                   * abs(singlet_character[i] - singlet_character[j]))
    if scores.max() > 0:
        scores = scores / scores.max()
    return bin_centres_MHz, scores


# old, edited: raises instead of padding with 5, 10, ... MHz
def action_spectrum_seeds(bin_centres_MHz, scores, number_of_seeds, min_separation_MHz):
    seeds_MHz = []
    for index in np.argsort(scores)[::-1]:
        frequency_MHz = bin_centres_MHz[index]
        if all(abs(frequency_MHz - seed) >= min_separation_MHz for seed in seeds_MHz):
            seeds_MHz.append(frequency_MHz)
        if len(seeds_MHz) == number_of_seeds:
            return np.array(seeds_MHz)
    raise ValueError(f"Only {len(seeds_MHz)} bins are far enough apart; asked for {number_of_seeds} seeds")


# new
def random_starting_frequencies(random_seed, number_of_frequencies, max_frequency_MHz):
    return np.random.default_rng(random_seed).uniform(0.0, max_frequency_MHz, size=number_of_frequencies)


# old, edited: engine yields instead of QuTiP
def disruption_cost(hypothesis, drive, times_us, max_step_us, orientations=None):
    singlet_yield, _ = yields(hypothesis, drive_rad_per_us(drive), times_us, max_step_us, orientations)
    return singlet_yield


# old, edited: both pairs in one pass
def discrimination_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations=None):
    # -|Y_S,A - Y_S,B|
    (singlet_yield_A, _), (singlet_yield_B, _) = yields_together([hypothesis_A, hypothesis_B],
                                                                 drive_rad_per_us(drive), times_us, max_step_us,
                                                                 orientations)
    return -abs(singlet_yield_A - singlet_yield_B)


# new
def bhattacharyya_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations=None):
    # -√B; the root keeps the scale like the yield gap, so the Nelder-Mead tolerances still make sense
    (singlet_yield_A, triplet_yield_A), (singlet_yield_B, triplet_yield_B) = yields_together(
        [hypothesis_A, hypothesis_B], drive_rad_per_us(drive), times_us, max_step_us, orientations)
    return -np.sqrt(bhattacharyya_distance(outcome_probabilities(singlet_yield_A, triplet_yield_A),
                                           outcome_probabilities(singlet_yield_B, triplet_yield_B)))


# new
def contrast_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, field_directions):
    # objective of the contrast runs: minimise -|C_A - C_B|, both pairs under the same drive
    contrast_A, contrast_B = (recombination_contrast(hypothesis, drive_rad_per_us(drive), times_us, max_step_us,
                                                     field_directions) for hypothesis in (hypothesis_A, hypothesis_B))
    return -abs(contrast_A - contrast_B)


# new
@dataclass(frozen=True)
class OptimisationResult:
    frequencies_MHz: np.ndarray
    raw_amplitudes: np.ndarray
    phases_rad: np.ndarray
    cost_history: np.ndarray  # [start, best after round 1, ...]
    accepted: np.ndarray
    evaluations: np.ndarray
    nelder_mead_costs: np.ndarray  # every cost call, all rounds in order


# new
@dataclass
class SearchState:
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


# new
def new_search_state(cost, seed_frequencies_MHz, random_seed):
    number_of_terms = len(seed_frequencies_MHz)
    frequencies_MHz = np.array(seed_frequencies_MHz, dtype=float)
    raw_amplitudes, phases_rad = np.ones(number_of_terms), np.zeros(number_of_terms)
    start_cost = cost(frequencies_MHz, raw_amplitudes, phases_rad)
    random_numbers = np.random.RandomState(random_seed)  # same generator as the old np.random.seed, so same nudges
    return SearchState(frequencies_MHz, raw_amplitudes, phases_rad, [start_cost], [], [], [], 0, random_numbers, 0.0)


# old, edited: one round split out of the loop, records every cost
def one_super_iteration(state, cost, frequency_perturbation_MHz, max_nelder_mead_iterations):
    # nudge every frequency, tune amplitudes and phases by Nelder-Mead, keep only if the cost went down
    number_of_terms = len(state.frequencies_MHz)
    nudge_MHz = state.random_numbers.uniform(-frequency_perturbation_MHz, frequency_perturbation_MHz,
                                             size=number_of_terms)
    candidate_frequencies_MHz = np.abs(state.frequencies_MHz + nudge_MHz)

    def cost_at_candidate(amplitudes_then_phases):
        value = cost(candidate_frequencies_MHz, amplitudes_then_phases[:number_of_terms],
                     amplitudes_then_phases[number_of_terms:])
        state.nelder_mead_costs.append(value)
        return value

    result = scipy.optimize.minimize(  # tolerances as in last year's runs
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


# new
def save_search_state(path, state):
    # the generator is stored as its internal state so a resumed run draws exactly the same nudges
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


# new
def load_search_state(path):
    saved = read_npz(path)
    if "nelder_mead_costs" not in saved:  # version 1 checkpoints did not record them
        saved["nelder_mead_costs"] = np.full(int(np.sum(saved["evaluations"])), np.nan)
    random_numbers = np.random.RandomState()
    random_numbers.set_state(("MT19937", saved["generator_keys"], saved["generator_position"],
                              saved["generator_has_gauss"], saved["generator_cached_gaussian"]))
    return SearchState(saved["frequencies_MHz"], saved["raw_amplitudes"], saved["phases_rad"],
                       list(saved["cost_history"]), list(saved["accepted"]), list(saved["evaluations"]),
                       list(saved["nelder_mead_costs"]), saved["super_iterations_without_improvement"],
                       random_numbers, saved["elapsed_seconds"])


# old, edited: patience, checkpoints and resume
def optimise_drive(cost, seed_frequencies_MHz, super_iterations, frequency_perturbation_MHz,
                   max_nelder_mead_iterations, random_seed, patience_super_iterations, checkpoint_path=None):
    """dCRAB-inspired greedy search over frequencies, raw amplitudes and phases; starts at amplitudes 1, phases 0."""
    label = "" if checkpoint_path is None else f"[{Path(checkpoint_path).stem}] "
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = load_search_state(checkpoint_path)
        print(f"  {label}resuming: {len(state.accepted)} super-iterations already done")
    else:
        state = new_search_state(cost, seed_frequencies_MHz, random_seed)
        print(f"  {label}start cost = {state.cost_history[0]:.6f}", flush=True)
    start_time = time.perf_counter() - state.elapsed_seconds

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
