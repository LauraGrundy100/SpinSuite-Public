"""Propagation of a hypothesis under a drive, and singlet and triplet yields.

Units: Hamiltonians and the drive in rad/µs, times in µs, rates in µs⁻¹ (see model.py).
The drive is given as f(t) in rad/µs, the coefficient of S1x + S2x; building and normalising
drives is control.py's job.

Method:
    State   ρ0 is split into weighted kets, each evolved by the non-Hermitian H − iK/2.
            This is exact for Haberkorn recombination, which keeps pure states pure.
    Drive   exp(−iθ(S1x + S2x)) has a closed form, because S1x + S2x has eigenvalues 1, 0, 0, −1.
    Steps   Strang splitting: half a drive kick, a full static step, another half kick (error ∝ step²).
    Yields  Richardson extrapolation over two step sizes cancels the step² error (error ∝ step⁴).
    Pairs   Several hypotheses under the same drive are evolved in one pass, as one larger system made of
            independent blocks (block-diagonal matrices: zeros between blocks, so they never interact).
            Stepping costs about the same for one block or two, so this nearly halves discrimination time.
    Average Yields averaged over orientations of the molecule (every direction equally likely), as a weighted sum
            over a fixed set of directions (Lebedev grid). All directions of one hypothesis are stepped together
            (batched: one NumPy call per step for every direction), about 9 times faster than one at a time.
            The one-at-a-time version is the reference in tests/reference_engine.py.
    Contrast For a molecule with no symmetry axis: yields for a list of field directions in the molecule's frame,
            the recombination yield of Daniel's scheme, and the contrast (highest minus lowest over the directions).
    Metric  Bhattacharyya distance between two hypotheses' outcome probabilities under counting noise
            (end of file): how quickly counting pairs tells the hypotheses apart.
"""

import numpy as np
import scipy.integrate
import scipy.linalg

from model import (drive_operator, effective_hamiltonian, field_direction_hypothesis, nuclear_dimension,
                   oriented_hypothesis, recombination_projector)


def initial_kets(hypothesis):
    """Split the initial state into weighted kets, ρ0 = Σ_j weights[j] |kets[:, j]⟩⟨kets[:, j]| (dimensionless).

    Each eigenvector of the 4×4 electron state with nonzero weight is paired with every nuclear
    basis state, since the nuclei start fully mixed. A singlet start gives one ket per nuclear state,
    as in the old code. Returns weights (length r) and kets (columns of a dimension × r array).
    """
    electron_weights, electron_kets = np.linalg.eigh(hypothesis.electron_initial_state)
    if np.any(electron_weights < -1e-12):
        raise ValueError(f"Electron initial state has negative eigenvalues: {electron_weights}")
    dimension = nuclear_dimension(hypothesis)
    weights, kets = [], []
    for electron_weight, electron_ket in zip(electron_weights, electron_kets.T):
        if electron_weight > 1e-12:  # drop states that are not populated
            for nuclear_state in np.eye(dimension):
                weights.append(electron_weight / dimension)
                kets.append(np.kron(electron_ket, nuclear_state))
    return np.array(weights), np.array(kets).T


def drive_exponential(drive_operator, drive_operator_squared, angle_rad):
    """exp(−iθX) for X = S1x + S2x and θ = angle_rad, by the closed form 1 − i sin θ X + (cos θ − 1) X².

    Valid only because X has eigenvalues 1, 0, −1, so X³ = X: every odd power of X is X and every
    even power (from 2 up) is X², and the exponential series sums to sines and cosines.
    Physically, it rotates both electron spins about x by θ. X² is passed in, so a caller that
    needs many steps computes it once, not on every step.
    """
    identity = np.eye(len(drive_operator))
    return (identity - 1j * np.sin(angle_rad) * drive_operator
            + (np.cos(angle_rad) - 1) * drive_operator_squared)


def populations_together(hypotheses, drive_rad_per_us, times_us, substeps):
    """Singlet and triplet populations Tr(P_S ρ), Tr(P_T ρ) at each time in times_us, for several hypotheses
    under the same drive, by Strang splitting. Returns one (singlet, triplet) pair of arrays per hypothesis.
    "Singlet" means the recombination state P_R (model.recombination_projector), which is P_S unless the hypothesis
    sets another, and "triplet" everything else; the same holds for every singlet/triplet quantity in this file.

    All hypotheses are evolved together as one block-diagonal system, so they share every step. Each output
    interval is split into `substeps` equal steps of length h. One step is D V D, with V = exp(−i(H − iK/2)h)
    and D = exp(−iθX/2), θ = f(t + h/2)·h (drive sampled mid-step). drive_rad_per_us: f(t) in rad/µs,
    accepting an array of times, or None. times_us must be evenly spaced; ρ0 applies at times_us[0]. Error ∝ h².
    """
    output_step_us = times_us[1] - times_us[0]
    if not np.allclose(np.diff(times_us), output_step_us, rtol=1e-9, atol=0):
        raise ValueError("times_us must be evenly spaced")
    step_us = output_step_us / substeps
    static_step = scipy.linalg.block_diag(
        *[scipy.linalg.expm(-1j * effective_hamiltonian(hypothesis) * step_us) for hypothesis in hypotheses])
    operator = scipy.linalg.block_diag(*[drive_operator(hypothesis) for hypothesis in hypotheses])
    operator_squared = operator @ operator  # block-diagonal X still has eigenvalues 1, 0, −1, so X³ = X holds
    singlet = scipy.linalg.block_diag(*[recombination_projector(hypothesis) for hypothesis in hypotheses])
    weights_and_kets = [initial_kets(hypothesis) for hypothesis in hypotheses]
    kets = scipy.linalg.block_diag(*[hypothesis_kets for _, hypothesis_kets in weights_and_kets])

    # Half-kick angle θ/2 for every step, from the drive at each step's midpoint.
    number_of_steps = (len(times_us) - 1) * substeps
    midpoints_us = times_us[0] + (np.arange(number_of_steps) + 0.5) * step_us
    if drive_rad_per_us is None:
        half_angles_rad = np.zeros(number_of_steps)
    else:
        half_angles_rad = 0.5 * drive_rad_per_us(midpoints_us) * step_us

    # Population of each ket (column) at each time; combined with weights per hypothesis at the end.
    singlet_by_ket = np.zeros((len(times_us), kets.shape[1]))
    total_by_ket = np.zeros((len(times_us), kets.shape[1]))
    for i in range(len(times_us)):
        if i > 0:  # advance from times_us[i - 1] to times_us[i]
            for step in range((i - 1) * substeps, i * substeps):
                half_kick = drive_exponential(operator, operator_squared, half_angles_rad[step])
                kets = half_kick @ (static_step @ (half_kick @ kets))
        singlet_by_ket[i] = np.sum(np.abs(singlet @ kets) ** 2, axis=0)
        total_by_ket[i] = np.sum(np.abs(kets) ** 2, axis=0)

    results, first_column = [], 0
    for weights, _ in weights_and_kets:
        columns = slice(first_column, first_column + len(weights))
        singlet_population = singlet_by_ket[:, columns] @ weights
        results.append((singlet_population, total_by_ket[:, columns] @ weights - singlet_population))
        first_column += len(weights)
    return results


def populations(hypothesis, drive_rad_per_us, times_us, substeps):
    """Singlet and triplet populations for one hypothesis: populations_together with a single block."""
    return populations_together([hypothesis], drive_rad_per_us, times_us, substeps)[0]


def yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations=None):
    """Singlet and triplet yields Y_S = k_S ∫ Tr(P_S ρ) dt, Y_T = k_T ∫ Tr(P_T ρ) dt (dimensionless, trapezoid on
    times_us) for several hypotheses under the same drive, in one pass. Returns one (Y_S, Y_T) per hypothesis.

    Populations are found with n substeps per output interval (n chosen so each step is at most
    max_step_us, in µs) and with 2n. Their errors are c·h² and c·h²/4, so (4 × fine − coarse)/3
    cancels the h² term (Richardson extrapolation), leaving an error ∝ h⁴.
    orientations: None (default) for the molecules as given; or (polar angles, azimuths, weights) from
    orientation_grid, to average over orientations instead (averaged_yields_together).
    """
    if orientations is not None:
        return averaged_yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations)
    output_step_us = times_us[1] - times_us[0]
    substeps = int(np.ceil(output_step_us / max_step_us - 1e-9))  # small allowance for rounding
    coarse = populations_together(hypotheses, drive_rad_per_us, times_us, substeps)
    fine = populations_together(hypotheses, drive_rad_per_us, times_us, 2 * substeps)
    results = []
    for hypothesis, (coarse_singlet, coarse_triplet), (fine_singlet, fine_triplet) in zip(hypotheses, coarse, fine):
        singlet_population = (4 * fine_singlet - coarse_singlet) / 3
        triplet_population = (4 * fine_triplet - coarse_triplet) / 3
        results.append((hypothesis.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us),
                        hypothesis.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us)))
    return results


def yields(hypothesis, drive_rad_per_us, times_us, max_step_us, orientations=None):
    """Singlet and triplet yields (Y_S, Y_T) for one hypothesis: yields_together with a single block.
    orientations: None for the molecule as given, or a grid from orientation_grid to average over orientations."""
    return yields_together([hypothesis], drive_rad_per_us, times_us, max_step_us, orientations)[0]


# Averaging over orientations
#
# In a sample of fixed molecules pointing every way, the measured yield is the average over all directions of the
# molecule's axis. A Lebedev grid is a fixed set of directions with weights, chosen so that the weighted sum equals
# the average over the sphere exactly for smooth functions up to a given order (like Simpson's rule, on a sphere).

def orientation_grid(lebedev_order):
    """Directions and weights for averaging over orientations, every direction equally likely (SciPy's Lebedev rule).

    Returns polar angles θ and azimuths φ (rad) of the molecule's axis, and weights summing to 1.
    lebedev_order: one of SciPy's orders, e.g. 11 (50 directions), 17 (110), 23 (194), 41 (590).
    """
    points, weights = scipy.integrate.lebedev_rule(lebedev_order)
    polar_angles_rad = np.arccos(np.clip(points[2], -1.0, 1.0))
    azimuths_rad = np.arctan2(points[1], points[0])
    return polar_angles_rad, azimuths_rad, weights / np.sum(weights)


def orientation_populations(oriented_hypotheses, drive_rad_per_us, times_us, substeps):
    """Singlet and triplet populations, each of shape (time, direction), for copies of one hypothesis turned to
    different directions, all stepped together under the same drive by Strang splitting (as populations_together).

    Turning a molecule changes neither the drive operator nor the number of starting kets, so every copy's kets stack
    into one (direction, dimension, kets) array. Each matrix product below then acts on all directions at once: the
    drive kick (the same for all), the static steps and the recombination projectors (one per direction; the
    projector turns with the molecule unless it is the singlet).
    """
    step_us = (times_us[1] - times_us[0]) / substeps
    static_steps = scipy.linalg.expm(  # shape (direction, dimension, dimension)
        -1j * np.array([effective_hamiltonian(hypothesis) for hypothesis in oriented_hypotheses]) * step_us)
    operator = drive_operator(oriented_hypotheses[0])
    operator_squared = operator @ operator
    singlet = np.array([recombination_projector(hypothesis) for hypothesis in oriented_hypotheses])
    weights_and_kets = [initial_kets(hypothesis) for hypothesis in oriented_hypotheses]
    weights = np.array([ket_weights for ket_weights, _ in weights_and_kets])  # (direction, kets)
    kets = np.array([direction_kets for _, direction_kets in weights_and_kets])  # (direction, dimension, kets)

    number_of_steps = (len(times_us) - 1) * substeps
    midpoints_us = times_us[0] + (np.arange(number_of_steps) + 0.5) * step_us
    if drive_rad_per_us is None:
        half_angles_rad = np.zeros(number_of_steps)
    else:
        half_angles_rad = 0.5 * drive_rad_per_us(midpoints_us) * step_us

    singlet_population = np.zeros((len(times_us), len(oriented_hypotheses)))
    total_population = np.zeros((len(times_us), len(oriented_hypotheses)))
    for i in range(len(times_us)):
        if i > 0:  # advance from times_us[i - 1] to times_us[i]
            for step in range((i - 1) * substeps, i * substeps):
                half_kick = drive_exponential(operator, operator_squared, half_angles_rad[step])
                kets = half_kick @ (static_steps @ (half_kick @ kets))
        # Σ over dimension of |amplitude|², then weighted sum over kets: one population per direction.
        singlet_population[i] = np.einsum("dak,dk->d", np.abs(singlet @ kets) ** 2, weights)
        total_population[i] = np.einsum("dak,dk->d", np.abs(kets) ** 2, weights)
    return singlet_population, total_population - singlet_population


def averaged_yields_together(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations):
    """Orientation-averaged yields (Y_S, Y_T), one pair per hypothesis, all under the same drive (dimensionless).

    orientations: (polar angles, azimuths, weights) from orientation_grid. Each hypothesis is turned to every
    direction (oriented_hypothesis in model.py) and all directions are stepped together (orientation_populations),
    with Richardson extrapolation as in yields_together. Per-direction yields are summed with the weights.
    Hypotheses go one at a time, so they may differ in size or number of starting kets.
    """
    polar_angles_rad, azimuths_rad, weights = orientations
    results = []
    for hypothesis in hypotheses:
        oriented = [oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
                    for polar_angle_rad, azimuth_rad in zip(polar_angles_rad, azimuths_rad)]
        singlet_yields, triplet_yields = turned_copies_yields(oriented, drive_rad_per_us, times_us, max_step_us)
        results.append((weights @ singlet_yields, weights @ triplet_yields))
    return results


def turned_copies_yields(turned_hypotheses, drive_rad_per_us, times_us, max_step_us):
    """Singlet and triplet yields, two arrays with one entry per copy, for copies of one hypothesis turned to
    different directions, all stepped together (orientation_populations) with Richardson extrapolation as in
    yields_together. All copies must share the rates (they do: turning does not change them)."""
    output_step_us = times_us[1] - times_us[0]
    substeps = int(np.ceil(output_step_us / max_step_us - 1e-9))  # as in yields_together
    coarse_singlet, coarse_triplet = orientation_populations(turned_hypotheses, drive_rad_per_us, times_us, substeps)
    fine_singlet, fine_triplet = orientation_populations(turned_hypotheses, drive_rad_per_us, times_us, 2 * substeps)
    singlet_population = (4 * fine_singlet - coarse_singlet) / 3  # shape (time, direction)
    triplet_population = (4 * fine_triplet - coarse_triplet) / 3
    rates = turned_hypotheses[0]
    return (rates.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us, axis=0),
            rates.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us, axis=0))


# Field directions in the molecule's frame, and the contrast
#
# For a molecule with no symmetry axis (flavin_tryptophan_hypothesis), the field direction is given in the molecule's
# own frame (model.field_direction_hypothesis). The contrast is how much the yield changes with direction: the
# highest minus the lowest yield over a list of directions, the quantity a compass depends on.

def field_direction_yields(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions):
    """Singlet and triplet yields (Y_S, Y_T), two arrays with one entry per field direction (dimensionless).

    field_directions: (polar angles, azimuths) in rad, in the molecule's frame. All directions are stepped together.
    """
    polar_angles_rad, azimuths_rad = field_directions
    turned = [field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
              for polar_angle_rad, azimuth_rad in zip(polar_angles_rad, azimuths_rad)]
    return turned_copies_yields(turned, drive_rad_per_us, times_us, max_step_us)


def recombination_yield(hypothesis, singlet_yield):
    """Yield of the recombination product alone, k_r ∫ Tr(P_R ρ) dt with k_r = k_S − k_T (dimensionless).

    For Daniel's scheme K = k_r P_R + k_f × identity (flavin_tryptophan_hypothesis: k_S = k_r + k_f, k_T = k_f).
    The engine's singlet_yield k_S ∫ Tr(P_R ρ) dt also counts the forward reaction (rate k_f) from the
    recombination state, so this keeps the k_r share only. The forward yield is Y_S + Y_T − this.
    Works on single values or arrays. Is 0 when k_S = k_T (every run before the flavin-tryptophan pair).
    """
    rate_through_projector = hypothesis.singlet_recombination_rate_per_us
    return (rate_through_projector - hypothesis.triplet_recombination_rate_per_us) / rate_through_projector \
        * np.asarray(singlet_yield)


def recombination_contrast(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions):
    """Contrast: highest minus lowest recombination yield over field_directions (dimensionless).

    field_directions: (polar angles, azimuths) in rad, in the molecule's frame. Daniel's primary metric (9 Oct).
    """
    singlet_yields, _ = field_direction_yields(hypothesis, drive_rad_per_us, times_us, max_step_us, field_directions)
    recombination_yields = recombination_yield(hypothesis, singlet_yields)
    return np.max(recombination_yields) - np.min(recombination_yields)


# Noise-aware discrimination (counting noise)
#
# Each pair ends in one of three outcomes by the end of the time window: singlet product, triplet product, or not
# yet recombined. Counting N pairs gives counts that wobble like coin flips (binomial, or multinomial for three
# outcomes). The Bhattacharyya distance B measures how far apart the outcome probabilities of two hypotheses are:
# after N pairs, the chance of picking the wrong hypothesis (equal prior odds) is at most ½ exp(−N B).

def outcome_probabilities(singlet_yield, triplet_yield):
    """Probabilities of the three outcomes per pair: (singlet product, triplet product, not yet recombined).

    The third is 1 − Y_S − Y_T. Engine error can make it slightly negative when it should be about 0, so values
    down to −1e-6 (well beyond the engine's error) are set to 0; anything more negative is a bug and is rejected.
    """
    unrecombined = 1.0 - singlet_yield - triplet_yield
    if unrecombined < -1e-6:
        raise ValueError(f"Y_S + Y_T = {singlet_yield + triplet_yield} exceeds 1")
    return np.array([singlet_yield, triplet_yield, max(unrecombined, 0.0)])


def bhattacharyya_distance(outcome_probabilities_A, outcome_probabilities_B):
    """Bhattacharyya distance B = −ln Σ √(p_A p_B) per pair (dimensionless, natural log) between two outcome lists.

    Σ √(p_A p_B) is the overlap of the two lists: 1 if identical (B = 0), 0 if they share no outcome (B infinite).
    For small differences, B ≈ (1/8) Σ (p_A − p_B)² / p.
    """
    overlap = np.sum(np.sqrt(np.asarray(outcome_probabilities_A) * np.asarray(outcome_probabilities_B)))
    if overlap == 0.0:
        return np.inf
    return max(-np.log(overlap), 0.0)  # rounding can put the overlap a hair above 1; B is never negative


def pairs_needed(distance_per_pair, error_probability):
    """Number of pairs N after which the chance of picking the wrong hypothesis is at most error_probability.

    distance_per_pair is the Bhattacharyya distance B. From ½ exp(−N B) ≤ error_probability,
    N = ln(1 / (2 × error_probability)) / B, rounded up. This is a guarantee (an upper bound on the error),
    so the true number needed can be smaller.
    """
    if distance_per_pair == 0.0:
        return np.inf
    return np.ceil(np.log(1.0 / (2.0 * error_probability)) / distance_per_pair)
