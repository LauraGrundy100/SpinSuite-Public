import numpy as np
import scipy.integrate
import scipy.linalg

from model import (drive_operator, effective_hamiltonian, field_direction_hypothesis, nuclear_dimension,
                   oriented_hypothesis, recombination_projector)

# Propagation is all numpy/scipy, no QuTiP: scipy.linalg.expm for the static step, my own closed-form drive kick,
# Strang splitting per step, Richardson extrapolation over two step sizes. Kets are evolved under H - iK/2.
# "singlet" below always means the recombination state P_R (the CISS one for pair B), "triplet" everything else.


# old, edited: any mixed electron start, not only the singlet
def initial_kets(hypothesis):
    # ρ0 = Σ w_j |ψ_j⟩⟨ψ_j|: each populated electron eigenstate times each nuclear basis state (nuclei fully mixed)
    electron_weights, electron_kets = np.linalg.eigh(hypothesis.electron_initial_state)
    if np.any(electron_weights < -1e-12):
        raise ValueError(f"Electron initial state has negative eigenvalues: {electron_weights}")
    dimension = nuclear_dimension(hypothesis)
    weights, kets = [], []
    for electron_weight, electron_ket in zip(electron_weights, electron_kets.T):
        if electron_weight > 1e-12:
            for nuclear_state in np.eye(dimension):
                weights.append(electron_weight / dimension)
                kets.append(np.kron(electron_ket, nuclear_state))
    return np.array(weights), np.array(kets).T


# new
def drive_exponential(drive_operator, drive_operator_squared, angle_rad):
    # exp(-iθX) = 1 - i sin θ X + (cos θ - 1) X², exact because X = S1x + S2x has eigenvalues 1, 0, -1 (X³ = X)
    identity = np.eye(len(drive_operator))
    return (identity - 1j * np.sin(angle_rad) * drive_operator
            + (np.cos(angle_rad) - 1) * drive_operator_squared)


# new
def populations_together(hypotheses, drive_rad_per_us, times_us, substeps):
    """Tr(P_R ρ) and Tr((1 - P_R) ρ) on times_us for several hypotheses under one drive, by Strang splitting."""
    # Hypotheses go in as independent blocks of one block-diagonal system, so they share every step.
    output_step_us = times_us[1] - times_us[0]
    if not np.allclose(np.diff(times_us), output_step_us, rtol=1e-9, atol=0):
        raise ValueError("times_us must be evenly spaced")
    step_us = output_step_us / substeps
    static_step = scipy.linalg.block_diag(
        *[scipy.linalg.expm(-1j * effective_hamiltonian(hypothesis) * step_us) for hypothesis in hypotheses])
    operator = scipy.linalg.block_diag(*[drive_operator(hypothesis) for hypothesis in hypotheses])
    operator_squared = operator @ operator
    singlet = scipy.linalg.block_diag(*[recombination_projector(hypothesis) for hypothesis in hypotheses])
    weights_and_kets = [initial_kets(hypothesis) for hypothesis in hypotheses]
    kets = scipy.linalg.block_diag(*[hypothesis_kets for _, hypothesis_kets in weights_and_kets])

    # drive sampled at each step's midpoint; half the kick either side of the static step
    number_of_steps = (len(times_us) - 1) * substeps
    midpoints_us = times_us[0] + (np.arange(number_of_steps) + 0.5) * step_us
    if drive_rad_per_us is None:
        half_angles_rad = np.zeros(number_of_steps)
    else:
        half_angles_rad = 0.5 * drive_rad_per_us(midpoints_us) * step_us

    singlet_by_ket = np.zeros((len(times_us), kets.shape[1]))
    total_by_ket = np.zeros((len(times_us), kets.shape[1]))
    for i in range(len(times_us)):
        if i > 0:
            for step in range((i - 1) * substeps, i * substeps):
                half_kick = drive_exponential(operator, operator_squared, half_angles_rad[step])
                kets = half_kick @ (static_step @ (half_kick @ kets))
        singlet_by_ket[i] = np.sum(np.abs(singlet @ kets) ** 2, axis=0)
        total_by_ket[i] = np.sum(np.abs(kets) ** 2, axis=0)  # norm decays through K

    results, first_column = [], 0
    for weights, _ in weights_and_kets:
        columns = slice(first_column, first_column + len(weights))
        singlet_population = singlet_by_ket[:, columns] @ weights
        results.append((singlet_population, total_by_ket[:, columns] @ weights - singlet_population))
        first_column += len(weights)
    return results


# new
def yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations=None):
    """Y_S = k_S ∫ Tr(P_R ρ) dt and Y_T = k_T ∫ Tr((1 - P_R) ρ) dt for each hypothesis (trapezoid on times_us)."""
    if orientations is not None:
        return averaged_yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations)
    # n substeps (step <= max_step_us) and 2n; (4 fine - coarse)/3 cancels the h² error of Strang splitting.
    # With the 3 ns grid and max_step_us = 0.00151 the steps are 1.5 ns and 0.75 ns.
    output_step_us = times_us[1] - times_us[0]
    substeps = int(np.ceil(output_step_us / max_step_us - 1e-9))
    coarse = populations_together(hypotheses, drive_rad_per_us, times_us, substeps)
    fine = populations_together(hypotheses, drive_rad_per_us, times_us, 2 * substeps)
    results = []
    for hypothesis, (coarse_singlet, coarse_triplet), (fine_singlet, fine_triplet) in zip(hypotheses, coarse, fine):
        singlet_population = (4 * fine_singlet - coarse_singlet) / 3
        triplet_population = (4 * fine_triplet - coarse_triplet) / 3
        results.append((hypothesis.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us),
                        hypothesis.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us)))
    return results


# new
def yields(hypothesis, drive_rad_per_us, times_us, max_step_us, orientations=None):
    return yields_together([hypothesis], drive_rad_per_us, times_us, max_step_us, orientations)[0]


# new
def orientation_grid(lebedev_order):
    # Lebedev directions and weights (sum 1) for averaging over a randomly oriented sample; off in these runs
    points, weights = scipy.integrate.lebedev_rule(lebedev_order)
    polar_angles_rad = np.arccos(np.clip(points[2], -1.0, 1.0))
    azimuths_rad = np.arctan2(points[1], points[0])
    return polar_angles_rad, azimuths_rad, weights / np.sum(weights)


# new
def orientation_populations(oriented_hypotheses, drive_rad_per_us, times_us, substeps):
    """As populations_together, for turned copies of one hypothesis, all directions stepped in one batched product."""
    # turning leaves the drive operator and the number of kets alone, so kets stack as (direction, dimension, ket)
    step_us = (times_us[1] - times_us[0]) / substeps
    static_steps = scipy.linalg.expm(
        -1j * np.array([effective_hamiltonian(hypothesis) for hypothesis in oriented_hypotheses]) * step_us)
    operator = drive_operator(oriented_hypotheses[0])
    operator_squared = operator @ operator
    singlet = np.array([recombination_projector(hypothesis) for hypothesis in oriented_hypotheses])
    weights_and_kets = [initial_kets(hypothesis) for hypothesis in oriented_hypotheses]
    weights = np.array([ket_weights for ket_weights, _ in weights_and_kets])
    kets = np.array([direction_kets for _, direction_kets in weights_and_kets])

    number_of_steps = (len(times_us) - 1) * substeps
    midpoints_us = times_us[0] + (np.arange(number_of_steps) + 0.5) * step_us
    if drive_rad_per_us is None:
        half_angles_rad = np.zeros(number_of_steps)
    else:
        half_angles_rad = 0.5 * drive_rad_per_us(midpoints_us) * step_us

    singlet_population = np.zeros((len(times_us), len(oriented_hypotheses)))
    total_population = np.zeros((len(times_us), len(oriented_hypotheses)))
    for i in range(len(times_us)):
        if i > 0:
            for step in range((i - 1) * substeps, i * substeps):
                half_kick = drive_exponential(operator, operator_squared, half_angles_rad[step])
                kets = half_kick @ (static_steps @ (half_kick @ kets))
        singlet_population[i] = np.einsum("dak,dk->d", np.abs(singlet @ kets) ** 2, weights)
        total_population[i] = np.einsum("dak,dk->d", np.abs(kets) ** 2, weights)
    return singlet_population, total_population - singlet_population


# new
def averaged_yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations):
    polar_angles_rad, azimuths_rad, weights = orientations
    results = []
    for hypothesis in hypotheses:
        oriented = [oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
                    for polar_angle_rad, azimuth_rad in zip(polar_angles_rad, azimuths_rad)]
        singlet_yields, triplet_yields = turned_copies_yields(oriented, drive_rad_per_us, times_us, max_step_us)
        results.append((weights @ singlet_yields, weights @ triplet_yields))
    return results


# new
def turned_copies_yields(turned_hypotheses, drive_rad_per_us, times_us, max_step_us):
    # one (Y_S, Y_T) per turned copy, Richardson as in yields_together; turning does not change the rates
    output_step_us = times_us[1] - times_us[0]
    substeps = int(np.ceil(output_step_us / max_step_us - 1e-9))
    coarse_singlet, coarse_triplet = orientation_populations(turned_hypotheses, drive_rad_per_us, times_us, substeps)
    fine_singlet, fine_triplet = orientation_populations(turned_hypotheses, drive_rad_per_us, times_us, 2 * substeps)
    singlet_population = (4 * fine_singlet - coarse_singlet) / 3
    triplet_population = (4 * fine_triplet - coarse_triplet) / 3
    rates = turned_hypotheses[0]
    return (rates.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us, axis=0),
            rates.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us, axis=0))


# new
def field_direction_yields(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions):
    # field_directions: (θ array, φ array) in rad, in the molecule's frame
    polar_angles_rad, azimuths_rad = field_directions
    turned = [field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
              for polar_angle_rad, azimuth_rad in zip(polar_angles_rad, azimuths_rad)]
    return turned_copies_yields(turned, drive_rad_per_us, times_us, max_step_us)


# new
def recombination_yield(hypothesis, singlet_yield):
    # Y_S = k_S ∫ Tr(P_R ρ) also counts the forward reaction out of P_R; keep the k_r = k_S - k_T share only
    rate_through_projector = hypothesis.singlet_recombination_rate_per_us
    return (rate_through_projector - hypothesis.triplet_recombination_rate_per_us) / rate_through_projector \
        * np.asarray(singlet_yield)


# new
def recombination_contrast(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions):
    # contrast C = highest minus lowest recombination yield over the given field directions (Daniel's measure)
    singlet_yields, _ = field_direction_yields(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions)
    recombination_yields = recombination_yield(hypothesis, singlet_yields)
    return np.max(recombination_yields) - np.min(recombination_yields)


# new
def outcome_probabilities(singlet_yield, triplet_yield):
    # (Y_S, Y_T, not yet reacted); small negative remainders from engine error are clipped to 0
    unrecombined = 1.0 - singlet_yield - triplet_yield
    if unrecombined < -1e-6:
        raise ValueError(f"Y_S + Y_T = {singlet_yield + triplet_yield} exceeds 1")
    return np.array([singlet_yield, triplet_yield, max(unrecombined, 0.0)])


# new
def bhattacharyya_distance(outcome_probabilities_A, outcome_probabilities_B):
    # B = -ln Σ √(p_A p_B) per counted pair; not used by the contrast runs
    overlap = np.sum(np.sqrt(np.asarray(outcome_probabilities_A) * np.asarray(outcome_probabilities_B)))
    if overlap == 0.0:
        return np.inf
    return max(-np.log(overlap), 0.0)


# new
def pairs_needed(distance_per_pair, error_probability):
    # from the bound P(wrong) <= ½ exp(-N B)
    if distance_per_pair == 0.0:
        return np.inf
    return np.ceil(np.log(1.0 / (2.0 * error_probability)) / distance_per_pair)
