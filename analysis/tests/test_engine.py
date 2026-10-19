"""Tests for engine.py, and for the density-matrix reference it is checked against."""

import numpy as np
import pytest
import qutip as qt
import scipy.linalg

from engine import (averaged_yields_together, bhattacharyya_distance, drive_exponential, field_direction_yields,
                    initial_kets, orientation_grid, outcome_probabilities, pairs_needed, populations,
                    recombination_contrast, recombination_yield, yields, yields_together)
from model import (Nucleus, angular_frequency, baseline_hypothesis, ciss_density_matrix, drive_operator,
                   field_direction_hypothesis, initial_density_matrix, oriented_hypothesis, singlet_density_matrix,
                   singlet_projector, triplet_density_matrix)
import reference_engine
from reference_engine import averaged_yields_one_direction_at_a_time, haberkorn_yields
from test_model import ciss_test_pair, flavin_tryptophan_test_pairs, static_haberkorn_yields


def two_tone_test_drive(times_us):
    """Fixed test drive f(t) in rad/µs: 5 MHz at 50 MHz plus 3 MHz at 11 MHz (arbitrary, not from any old run)."""
    return (angular_frequency(5.0) * np.sin(angular_frequency(50.0) * times_us)
            + angular_frequency(3.0) * np.sin(angular_frequency(11.0) * times_us + 0.4))


def general_hypothesis():
    """Baseline pair with a hydrogen added on radical 2, k_S ≠ k_T and a triplet start: nothing the old engine handles.

    The baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    """
    baseline = baseline_hypothesis(0.3)
    hydrogen = Nucleus(radical=2, spin=0.5, hyperfine_tensor_MHz=5.0 * np.eye(3))
    return baseline.variant(nuclei=baseline.nuclei + (hydrogen,), electron_initial_state=triplet_density_matrix(),
                            singlet_recombination_rate_per_us=2.0, triplet_recombination_rate_per_us=0.5)


# %% Density-matrix reference (tests/reference_engine.py, last section)

def test_density_matrix_reference_matches_qutip_reference_with_equal_rates():
    # With k_S = k_T the two references solve the same problem by different routes (kets in QuTiP,
    # full ρ in solve_ivp). Baseline parameters and the 0–6 µs, 2000-point grid come from DISCO (old repo)
    # and may rely on stale placeholder parameters. Measured agreement: about 8e-11.
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, _, _ = haberkorn_yields(baseline_hypothesis(0.3), two_tone_test_drive, times_us)
    operators = reference_engine.spin_operators()
    hamiltonian = [reference_engine.static_hamiltonian(operators, 28.0, 0.3, 1.0, 28.0),
                   [operators["S1x"] + operators["S2x"], two_tone_test_drive]]
    population = reference_engine.singlet_population(hamiltonian, times_us)
    assert abs(singlet_yield - reference_engine.singlet_yield(population, times_us, 1.0)) < 1e-6


def test_density_matrix_reference_matches_exact_static_propagator():
    # No drive: the static propagator in test_model.py is exact (one expm), so this checks the general
    # reference for k_S ≠ k_T, a triplet start and two nuclei. Measured agreement: about 9e-13.
    times_us = np.linspace(0, 6, 2000)
    hypothesis = general_hypothesis()
    singlet_yield, triplet_yield, remaining = haberkorn_yields(hypothesis, None, times_us)
    exact_singlet, exact_triplet, exact_remaining = static_haberkorn_yields(hypothesis, times_us)
    assert abs(singlet_yield - exact_singlet) < 1e-6
    assert abs(triplet_yield - exact_triplet) < 1e-6
    assert abs(remaining - exact_remaining) < 1e-6


def test_density_matrix_reference_conserves_probability_with_drive():
    # Y_S + Y_T + Tr ρ(T) = 1, up to trapezoid error, which is small on this fine grid. Measured: about 2e-9.
    times_us = np.linspace(0, 6, 20000)
    singlet_yield, triplet_yield, remaining = haberkorn_yields(general_hypothesis(), two_tone_test_drive, times_us)
    assert abs(singlet_yield + triplet_yield + remaining - 1) < 1e-6


# %% Drive operator, initial kets and the drive exponential

def test_drive_operator_matches_reference_engine():
    reference_operators = reference_engine.spin_operators()
    reference = (reference_operators["S1x"] + reference_operators["S2x"]).full()
    np.testing.assert_allclose(drive_operator(baseline_hypothesis(0.0)), reference, atol=1e-15)


def test_drive_operator_eigenvalues_and_singlet():
    # Eigenvalues 1, 0, 0, −1, each three times (one per nuclear state). X³ = X is what the closed form
    # needs; X P_S = 0 means the drive never mixes singlet and triplet, so it commutes with K.
    hypothesis = baseline_hypothesis(0.3)
    operator = drive_operator(hypothesis)
    np.testing.assert_allclose(np.linalg.eigvalsh(operator), [-1] * 3 + [0] * 6 + [1] * 3, atol=1e-12)
    np.testing.assert_allclose(operator @ operator @ operator, operator, atol=1e-12)
    np.testing.assert_allclose(operator @ singlet_projector(hypothesis), 0, atol=1e-12)


def mixed_electron_state():
    """60% singlet, 40% triplet mixture: a full-rank electron state, for tests."""
    return 0.6 * singlet_density_matrix() + 0.4 * triplet_density_matrix()


@pytest.mark.parametrize("hypothesis, expected_number_of_kets", [
    (baseline_hypothesis(0.0), 3),                                                         # singlet: 1 × 3 nuclear states
    (baseline_hypothesis(0.0).variant(electron_initial_state=triplet_density_matrix()), 9),  # 3 triplets × 3
    (baseline_hypothesis(0.0).variant(electron_initial_state=mixed_electron_state()), 12),   # 4 × 3
    (general_hypothesis(), 18),                                                             # 3 triplets × 3 × 2
])
def test_initial_kets_rebuild_initial_density_matrix(hypothesis, expected_number_of_kets):
    weights, kets = initial_kets(hypothesis)
    assert len(weights) == kets.shape[1] == expected_number_of_kets
    assert np.isclose(np.sum(weights), 1.0)
    np.testing.assert_allclose(kets.conj().T @ kets, np.eye(expected_number_of_kets), atol=1e-12)  # orthonormal
    rebuilt = sum(weight * np.outer(ket, ket.conj()) for weight, ket in zip(weights, kets.T))
    np.testing.assert_allclose(rebuilt, initial_density_matrix(hypothesis), atol=1e-12)


def test_singlet_start_gives_old_kets():
    # The old code started from |S⟩ ⊗ |m⟩ for each nuclear state m, each weight 1/3 (up to an overall sign).
    weights, kets = initial_kets(baseline_hypothesis(0.0))
    np.testing.assert_allclose(weights, [1 / 3] * 3)
    for nuclear_state in range(3):
        old_ket = qt.tensor(reference_engine.electron_singlet(), qt.basis(3, nuclear_state)).full().ravel()
        assert np.isclose(abs(np.vdot(old_ket, kets[:, nuclear_state])), 1.0)


def test_unphysical_initial_state_is_rejected():
    not_a_state = singlet_density_matrix() - 0.5 * triplet_density_matrix()  # negative weights
    with pytest.raises(ValueError):
        initial_kets(baseline_hypothesis(0.0).variant(electron_initial_state=not_a_state))


@pytest.mark.parametrize("hypothesis", [baseline_hypothesis(0.0), general_hypothesis()])
@pytest.mark.parametrize("angle_rad", [0.0, 0.3, -1.2, np.pi, 7.5])
def test_drive_exponential_matches_expm(hypothesis, angle_rad):
    operator = drive_operator(hypothesis)
    closed_form = drive_exponential(operator, operator @ operator, angle_rad)
    np.testing.assert_allclose(closed_form, scipy.linalg.expm(-1j * angle_rad * operator), atol=1e-12)
    np.testing.assert_allclose(closed_form @ closed_form.conj().T, np.eye(len(operator)), atol=1e-12)  # unitary


# %% Propagation and yields

# Step limit for yields(). On the 0–6 µs, 2000-point grid it gives 5 and 10 substeps per output interval.
MAX_STEP_US = 0.00075

# Drives from DISCO (old repo); they may rely on stale placeholder parameters. Copied, not imported, from
# tests/test_reference_engine.py, which is temporary, while these engine tests are permanent.
# Optimised run-3 drive, from All DISCO/DISCO_run_1/run3_discriminate/run3_results.npz (frequencies converted to MHz):
RUN3_FREQUENCIES_MHz = [31.278903364378777, 28.331912643861415, 0.29370083696399624, 10.865639062914235]
RUN3_RAW_AMPLITUDES = [-0.360134557742187, 142.36189395265416, -0.34001651040795267, 2476.2233989893016]
RUN3_PHASES_RAD = [9.91228216421405, 0.8418697480570057, 0.012293007456405332, -0.1589099776003636]
# Monochromatic scan points, from Main Experimental scripts/PolyDiscrim_results/PolyDiscrim_results.npz:
MONOCHROMATIC_FREQUENCIES_MHz = [11.020433694745622, 49.97581317764804]
AMPLITUDE_CAP_MHz = 7.0  # 0.25 × A_iso, DISCO value


def disco_drives():
    """(label, drive function in rad/µs) for the run-3 drive and the two monochromatic points, L-infinity scaled as in DISCO."""
    drives = [("run3", reference_engine.drive_coefficient(RUN3_FREQUENCIES_MHz, RUN3_RAW_AMPLITUDES,
                                                          RUN3_PHASES_RAD, AMPLITUDE_CAP_MHz))]
    for frequency_MHz in MONOCHROMATIC_FREQUENCIES_MHz:
        drives.append((f"{frequency_MHz:.2f} MHz",
                       reference_engine.drive_coefficient([frequency_MHz], [1.0], [0.0], AMPLITUDE_CAP_MHz)))
    return drives


def unequal_rates(electron_initial_state):
    """Baseline pair (DISCO parameters, may be stale placeholders) at delta 0.3 with k_S = 2, k_T = 0.5 µs⁻¹."""
    return baseline_hypothesis(0.3).variant(electron_initial_state=electron_initial_state,
                                            singlet_recombination_rate_per_us=2.0,
                                            triplet_recombination_rate_per_us=0.5)


@pytest.mark.parametrize("hypothesis", [baseline_hypothesis(0.3), general_hypothesis()])
def test_no_drive_matches_exact_static_propagator(hypothesis):
    # With no drive every step is the exact static propagator, so only rounding error is left (measured: about 4e-14).
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, triplet_yield = yields(hypothesis, None, times_us, MAX_STEP_US)
    exact_singlet, exact_triplet, _ = static_haberkorn_yields(hypothesis, times_us)
    assert abs(singlet_yield - exact_singlet) < 1e-9
    assert abs(triplet_yield - exact_triplet) < 1e-9


@pytest.mark.parametrize("delta", [0.0, 0.3])
@pytest.mark.parametrize("drive_index", [0, 1, 2])
def test_equal_rates_match_qutip_reference_with_disco_drives(delta, drive_index):
    # k_S = k_T, so the old engine applies. Hypothesis, drives and grid come from DISCO (old repo) and may
    # rely on stale placeholder parameters. Measured agreement: 2e-9 to 5e-8.
    _, drive = disco_drives()[drive_index]
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, _ = yields(baseline_hypothesis(delta), drive, times_us, MAX_STEP_US)
    operators = reference_engine.spin_operators()
    hamiltonian = [reference_engine.static_hamiltonian(operators, 28.0, delta, 1.0, 28.0),
                   [operators["S1x"] + operators["S2x"], drive]]
    population = reference_engine.singlet_population(hamiltonian, times_us)
    assert abs(singlet_yield - reference_engine.singlet_yield(population, times_us, 1.0)) < 1e-6


@pytest.mark.parametrize("hypothesis", [
    unequal_rates(singlet_density_matrix()),
    unequal_rates(triplet_density_matrix()),
    general_hypothesis(),  # two nuclei, triplet start, k_S ≠ k_T
    # CISS start (a singlet-T0 superposition); baseline parameters from DISCO, may be stale placeholders.
    baseline_hypothesis(0.3).variant(electron_initial_state=ciss_density_matrix(0.25 * np.pi)),
    # The same CISS pair, molecule tilted (θ = 1.0, φ = 2.0 rad): hyperfine axis and CISS axis off the field.
    oriented_hypothesis(baseline_hypothesis(0.3).variant(electron_initial_state=ciss_density_matrix(0.25 * np.pi)),
                        1.0, 2.0),
])
def test_matches_density_matrix_reference(hypothesis):
    # Measured agreement: about 4e-8 for the first four (CISS start: 3.7e-8), in both yields; tilted: 8e-9.
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, triplet_yield = yields(hypothesis, two_tone_test_drive, times_us, MAX_STEP_US)
    reference_singlet, reference_triplet, _ = haberkorn_yields(hypothesis, two_tone_test_drive, times_us)
    assert abs(singlet_yield - reference_singlet) < 1e-6
    assert abs(triplet_yield - reference_triplet) < 1e-6


def test_strang_error_falls_as_step_squared():
    # Doubling the substeps should cut the plain Strang error by about 4 (this is what Richardson relies on).
    # Measured: 5.0e-5 at 4 substeps, 1.2e-5 at 8, ratio 3.98.
    times_us = np.linspace(0, 6, 2000)
    hypothesis = baseline_hypothesis(0.3)
    reference_singlet, _, _ = haberkorn_yields(hypothesis, two_tone_test_drive, times_us)
    errors = []
    for substeps in (4, 8):
        singlet_population, _ = populations(hypothesis, two_tone_test_drive, times_us, substeps)
        errors.append(abs(np.trapezoid(singlet_population, times_us) - reference_singlet))
    assert 3 < errors[0] / errors[1] < 5


def test_uneven_time_grid_is_rejected():
    uneven_times_us = np.array([0.0, 0.1, 0.25, 0.3])
    with pytest.raises(ValueError):
        populations(baseline_hypothesis(0.0), None, uneven_times_us, 1)


def test_several_hypotheses_in_one_pass_match_separate_passes():
    # One pass treats the hypotheses as independent blocks of one larger system, so each must get exactly
    # what it gets alone. Includes different sizes (12 and 24) and k_S ≠ k_T. Baseline parameters come from
    # DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 6, 2000)
    hypotheses = [baseline_hypothesis(0.0), general_hypothesis(), unequal_rates(singlet_density_matrix())]
    together = yields_together(hypotheses, two_tone_test_drive, times_us, 0.003)
    for hypothesis, (singlet_yield, triplet_yield) in zip(hypotheses, together):
        alone_singlet, alone_triplet = yields(hypothesis, two_tone_test_drive, times_us, 0.003)
        assert abs(singlet_yield - alone_singlet) < 1e-12
        assert abs(triplet_yield - alone_triplet) < 1e-12


# %% Noise-aware discrimination (Bhattacharyya distance)

def test_bhattacharyya_coin_example():
    # Worked example by hand: fair coin against a 60/40 coin. Overlap = √0.30 + √0.20, B = −ln(overlap) = 0.0050767.
    distance = bhattacharyya_distance([0.5, 0.5], [0.6, 0.4])
    assert distance == pytest.approx(0.0050767, abs=1e-7)
    assert pairs_needed(distance, 0.01) == 771  # ln(50) / B = 770.58, rounded up


def test_bhattacharyya_zero_for_identical_and_symmetric():
    probabilities_A, probabilities_B = [0.4, 0.5975, 0.0025], [0.35, 0.6, 0.05]
    assert bhattacharyya_distance(probabilities_A, probabilities_A) == pytest.approx(0.0, abs=1e-15)
    assert bhattacharyya_distance(probabilities_A, probabilities_B) == bhattacharyya_distance(probabilities_B,
                                                                                              probabilities_A)
    assert pairs_needed(0.0, 0.01) == np.inf


def test_bhattacharyya_infinite_when_outcomes_never_overlap():
    # A always gives singlet product, B never does: one pair is enough to tell them apart.
    assert bhattacharyya_distance([1.0, 0.0, 0.0], [0.0, 0.7, 0.3]) == np.inf


def test_bhattacharyya_small_difference_formula():
    # For small differences B ≈ (1/8) Σ (Δp)² / p. With two outcomes this is D² / (8 Y (1 − Y)).
    # Here p is taken at the midpoint of A and B, which makes the approximation accurate to fourth order.
    probabilities_A = np.array([0.4, 0.5975, 0.0025])
    probabilities_B = probabilities_A + 1e-3 * np.array([1.0, -0.99, -0.01])  # changes add to 0, so B sums to 1
    midpoint = (probabilities_A + probabilities_B) / 2
    approximation = np.sum((probabilities_A - probabilities_B) ** 2 / midpoint) / 8
    assert bhattacharyya_distance(probabilities_A, probabilities_B) == pytest.approx(approximation, rel=1e-4)


def test_merging_outcomes_never_increases_bhattacharyya():
    # Lumping triplet product and unrecombined into one bin throws information away, so B can only fall or stay.
    random_numbers = np.random.default_rng(20261008)
    for _ in range(200):
        probabilities_A, probabilities_B = random_numbers.dirichlet([1, 1, 1], size=2)
        merged_A = [probabilities_A[0], probabilities_A[1] + probabilities_A[2]]
        merged_B = [probabilities_B[0], probabilities_B[1] + probabilities_B[2]]
        assert bhattacharyya_distance(merged_A, merged_B) <= bhattacharyya_distance(probabilities_A,
                                                                                   probabilities_B) + 1e-15


def test_outcome_probabilities_sum_to_one_and_reject_impossible_yields():
    assert outcome_probabilities(0.4, 0.5) == pytest.approx([0.4, 0.5, 0.1])
    assert outcome_probabilities(0.4, 0.6 + 1e-9)[2] == 0.0  # engine-sized overshoot set to 0
    with pytest.raises(ValueError):
        outcome_probabilities(0.5, 0.6)


def test_unrecombined_fraction_with_equal_rates_is_exp_minus_k_t():
    # With k_S = k_T = k the population decays as exp(−k t) whatever the spins do, so after 6 µs the unrecombined
    # fraction is exp(−6 k). Baseline parameters come from DISCO (old repo) and may rely on stale placeholder
    # parameters. Tolerance covers the trapezoid rule on 2000 points (error about 1e-6).
    hypothesis = baseline_hypothesis(0.3)
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, triplet_yield = yields(hypothesis, two_tone_test_drive, times_us, MAX_STEP_US)
    rate_per_us = hypothesis.singlet_recombination_rate_per_us
    assert outcome_probabilities(singlet_yield, triplet_yield)[2] == pytest.approx(np.exp(-6 * rate_per_us), abs=1e-5)


# %% Averaging over orientations

def test_orientation_grid_averages_known_functions():
    # Over the sphere, the average of n_z² (and of n_x²) is 1/3, of n_z⁴ is 1/5, and of n_x n_z is 0. Rebuilding n
    # from θ and φ also checks the conversion from SciPy's Cartesian points.
    polar_angles_rad, azimuths_rad, weights = orientation_grid(17)
    n_x = np.sin(polar_angles_rad) * np.cos(azimuths_rad)
    n_z = np.cos(polar_angles_rad)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-14)
    assert np.all(weights > 0)
    assert weights @ n_z ** 2 == pytest.approx(1 / 3, abs=1e-14)
    assert weights @ n_x ** 2 == pytest.approx(1 / 3, abs=1e-14)
    assert weights @ n_z ** 4 == pytest.approx(1 / 5, abs=1e-14)
    assert weights @ (n_x * n_z) == pytest.approx(0.0, abs=1e-14)


def test_without_a_field_the_average_equals_the_aligned_yield():
    # With no field and no drive, orientation cannot matter (see tests/test_model.py), so the average is the
    # aligned value. Measured difference: 2e-14.
    times_us = np.linspace(0, 2, 401)
    no_field = ciss_test_pair().variant(zeeman_frequency_MHz=0.0)
    averaged = averaged_yields_together([no_field], None, times_us, 0.005, orientation_grid(5))[0]
    assert np.allclose(averaged, yields(no_field, None, times_us, 0.005), atol=1e-12)


def test_no_drive_average_matches_an_average_over_tilt():
    # With no drive only the tilt θ matters (spinning about the field is a symmetry), so an independent average is
    # ½ ∫ Y(θ) d(cos θ), done here by Gauss-Legendre over cos θ with the exact static propagator. Short window
    # (0–2 µs) to keep the test quick; with no drive the engine is exact at any step. Measured (50 directions
    # against 64 tilts): yields agree to 9e-5 and the gap Y_S,A − Y_S,B (0.018) to 3.5e-5. The yields vary
    # sharply with tilt, so both averages converge slowly; a wrong angle or weight would give errors near 0.01.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 2, 401)
    hypotheses = [baseline_hypothesis(0.3), ciss_test_pair()]
    averaged = np.array(averaged_yields_together(hypotheses, None, times_us, 0.005, orientation_grid(11)))
    cos_tilts, tilt_weights = np.polynomial.legendre.leggauss(64)
    over_tilt = np.array([sum(0.5 * weight * np.array(static_haberkorn_yields(
        oriented_hypothesis(hypothesis, np.arccos(cos_tilt), 0.0), times_us)[:2])
        for cos_tilt, weight in zip(cos_tilts, tilt_weights)) for hypothesis in hypotheses])
    assert np.max(np.abs(averaged - over_tilt)) < 2e-4
    assert abs((averaged[0, 0] - averaged[1, 0]) - (over_tilt[0, 0] - over_tilt[1, 0])) < 1e-4


def test_batched_average_matches_one_direction_at_a_time():
    # The batched engine and the plain loop (tests/reference_engine.py) do the same arithmetic in a different order,
    # so they must agree to rounding error. The third hypothesis has two nuclei and a triplet start, so the
    # hypotheses differ in size and number of starting kets. Short window to keep the test quick. Measured: 3e-16.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 1, 201)
    hypotheses = [baseline_hypothesis(0.3), ciss_test_pair(), general_hypothesis()]
    orientations = orientation_grid(5)
    batched = averaged_yields_together(hypotheses, two_tone_test_drive, times_us, 0.0015, orientations)
    one_at_a_time = averaged_yields_one_direction_at_a_time(hypotheses, two_tone_test_drive, times_us, 0.0015,
                                                             orientations)
    assert np.allclose(batched, one_at_a_time, rtol=0, atol=1e-12)


# %% Flavin-tryptophan pair (36 dimensions), field directions, recombination yield and contrast

TEST_FIELD_DIRECTIONS = (np.array([0.0, 1.0, 2.4]), np.array([0.0, 2.0, -1.1]))  # (θ, φ) in rad, molecule frame


@pytest.mark.parametrize("pair", [0, 1])  # classical, CISS (recombination through the CISS state, k_r = 10)
def test_flavin_tryptophan_pair_matches_density_matrix_reference(pair):
    # 36 dimensions, two nuclei on different radicals, full tensors, dipolar term, field off every tensor axis,
    # strong two-tone drive. Short window (0–2 µs, 3 ns grid) to keep the reference quick. Measured agreement:
    # 1.0e-7 (classical) and 1.2e-7 (CISS) in Y_S, 4e-8 and 6e-8 in Y_T.
    hypothesis = field_direction_hypothesis(flavin_tryptophan_test_pairs()[pair], 1.0, 2.0)
    times_us = np.linspace(0, 2, 667)
    singlet_yield, triplet_yield = yields(hypothesis, two_tone_test_drive, times_us, MAX_STEP_US)
    reference_singlet, reference_triplet, _ = haberkorn_yields(hypothesis, two_tone_test_drive, times_us)
    assert abs(singlet_yield - reference_singlet) < 1e-6
    assert abs(triplet_yield - reference_triplet) < 1e-6


def test_field_direction_yields_match_one_direction_at_a_time():
    # Batched over directions (each with its own turned CISS recombination state) against plain single runs.
    _, hypothesis = flavin_tryptophan_test_pairs()
    times_us = np.linspace(0, 1, 201)
    batched = field_direction_yields(hypothesis, two_tone_test_drive, times_us, 0.0015, TEST_FIELD_DIRECTIONS)
    for index, (polar_angle_rad, azimuth_rad) in enumerate(zip(*TEST_FIELD_DIRECTIONS)):
        alone = yields(field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad), two_tone_test_drive,
                       times_us, 0.0015)
        assert abs(batched[0][index] - alone[0]) < 1e-12
        assert abs(batched[1][index] - alone[1]) < 1e-12


def test_batched_average_turns_a_ciss_recombination_state():
    # As test_batched_average_matches_one_direction_at_a_time, with a recombination state that turns with the molecule.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 1, 201)
    hypothesis = ciss_test_pair().variant(recombination_state=ciss_density_matrix(-0.25 * np.pi),
                                          singlet_recombination_rate_per_us=3.0)
    batched = averaged_yields_together([hypothesis], two_tone_test_drive, times_us, 0.0015, orientation_grid(5))
    one_at_a_time = averaged_yields_one_direction_at_a_time([hypothesis], two_tone_test_drive, times_us, 0.0015,
                                                             orientation_grid(5))
    assert np.allclose(batched, one_at_a_time, rtol=0, atol=1e-12)


def test_recombination_yield_is_k_r_times_the_recombination_state_population():
    # k_r ∫ Tr(P_R ρ) dt straight from the density-matrix reference's populations. Short window.
    _, hypothesis = flavin_tryptophan_test_pairs()  # k_r = 10, k_f = 1
    hypothesis = field_direction_hypothesis(hypothesis, 0.6, 0.4)
    times_us = np.linspace(0, 2, 667)
    singlet_yield, _ = yields(hypothesis, None, times_us, 0.003)
    recombination_population, _ = reference_engine.haberkorn_populations(hypothesis, None, times_us)
    assert recombination_yield(hypothesis, singlet_yield) == pytest.approx(
        10.0 * np.trapezoid(recombination_population, times_us), abs=1e-6)
    assert recombination_yield(baseline_hypothesis(0.3), 0.4) == 0.0  # k_S = k_T: no separate recombination product


@pytest.mark.parametrize("pair", [0, 1])
def test_almost_nothing_is_left_unreacted_after_12_us(pair):
    # Recombination, forward reaction and the unreacted rest add to 1. Every pair reacts at least at k_f = 1 µs⁻¹, so
    # after 12 µs at most exp(−12) = 6e-6 is left; the trapezoid rule on a 3 ns grid adds about 3e-6 (measured).
    hypothesis = flavin_tryptophan_test_pairs()[pair]
    singlet_yield, triplet_yield = yields(hypothesis, None, np.linspace(0, 12, 4001), 0.003)
    forward_yield = singlet_yield + triplet_yield - recombination_yield(hypothesis, singlet_yield)
    unreacted = 1.0 - recombination_yield(hypothesis, singlet_yield) - forward_yield
    assert abs(unreacted) < 1e-5


def test_contrast_is_highest_minus_lowest_recombination_yield():
    classical, _ = flavin_tryptophan_test_pairs()
    times_us = np.linspace(0, 1, 201)
    singlet_yields, _ = field_direction_yields(classical, None, times_us, 0.005, TEST_FIELD_DIRECTIONS)
    recombination_yields = 0.5 * singlet_yields  # k_r / k_S = 1 / 2 for the classical pair
    assert recombination_contrast(classical, None, times_us, 0.005, TEST_FIELD_DIRECTIONS) == pytest.approx(
        np.max(recombination_yields) - np.min(recombination_yields), abs=1e-15)
    one_direction_twice = (np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert recombination_contrast(classical, None, times_us, 0.005, one_direction_twice) == 0.0
