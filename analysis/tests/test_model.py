"""Tests for model.py."""

import dataclasses

import numpy as np
import pytest
import qutip as qt
import scipy.linalg

from model import (Nucleus, angular_frequency, axial_hyperfine_tensor_MHz, baseline_hypothesis,
                   ciss_density_matrix, effective_hamiltonian, field_direction_hypothesis, flavin_tryptophan_hypothesis,
                   initial_density_matrix, oriented_hypothesis, recombination_operator, recombination_projector,
                   rotation_matrix, singlet_density_matrix, singlet_projector, spin_matrices, spin_operators,
                   static_hamiltonian, triplet_density_matrix, zeeman_frequency_MHz)
import reference_engine
from reference_engine import electron_singlet


# %% Units

def test_one_MHz_is_two_pi_rad_per_us():
    assert angular_frequency(1.0) == 2 * np.pi


def test_matches_old_repo_convention():
    # Old repo, All DISCO/DISCO.py lines 149-151: A_iso = 28 * 2 * np.pi, J = 1 * 2 * np.pi
    assert angular_frequency(28.0) == 28 * 2 * np.pi
    assert angular_frequency(1.0) == 1 * 2 * np.pi


def test_works_on_arrays():
    # Old repo, DISCO.py line 595: seeds_rad = seeds_MHz * 2 * np.pi
    seeds_MHz = np.array([11.0, 50.0, 63.5])
    np.testing.assert_array_equal(angular_frequency(seeds_MHz), seeds_MHz * 2 * np.pi)


# %% Hyperfine tensor and electron initial states

def test_axial_tensor_diagonal_and_trace():
    tensor = axial_hyperfine_tensor_MHz(10.0, 0.3)
    np.testing.assert_allclose(np.diag(tensor), [7.0, 7.0, 16.0])
    assert np.isclose(np.trace(tensor), 30.0)  # isotropic part 3A, whatever δ is


def test_singlet_density_matrix_matches_reference_engine():
    reference = qt.ket2dm(electron_singlet()).full()
    np.testing.assert_allclose(singlet_density_matrix(), reference, atol=1e-15)


@pytest.mark.parametrize("density_matrix", [singlet_density_matrix(), triplet_density_matrix(),
                                            ciss_density_matrix(0.1 * np.pi), ciss_density_matrix(-0.3 * np.pi)])
def test_initial_states_are_valid_density_matrices(density_matrix):
    assert np.isclose(np.trace(density_matrix), 1.0)
    np.testing.assert_allclose(density_matrix, density_matrix.conj().T)
    assert np.all(np.linalg.eigvalsh(density_matrix) > -1e-12)


# %% CISS start (Luo & Hore 2021, New J. Phys. 23 043032)

def test_ciss_limits_are_singlet_and_one_spin_up_one_down():
    np.testing.assert_allclose(ciss_density_matrix(0.0), singlet_density_matrix(), atol=1e-15)  # no CISS
    up_down, down_up = np.zeros((4, 4)), np.zeros((4, 4))
    up_down[1, 1] = down_up[2, 2] = 1.0  # basis order ↑↑, ↑↓, ↓↑, ↓↓
    np.testing.assert_allclose(ciss_density_matrix(np.pi / 2), up_down, atol=1e-15)  # 100% CISS
    np.testing.assert_allclose(ciss_density_matrix(-np.pi / 2), down_up, atol=1e-15)  # opposite chirality


def test_ciss_start_is_pure_with_singlet_fraction_cos_squared():
    # Worked example: χ = 0.1π gives singlet fraction cos²(0.05π) = 0.97553, the rest T0.
    density_matrix = ciss_density_matrix(0.1 * np.pi)
    np.testing.assert_allclose(density_matrix @ density_matrix, density_matrix, atol=1e-15)  # pure state
    assert abs(np.trace(singlet_density_matrix() @ density_matrix).real - 0.97553) < 1e-5


@pytest.mark.parametrize("ciss_angle_rad", [0.0, 0.05 * np.pi, 0.3 * np.pi, np.pi / 2, -0.2 * np.pi])
def test_ciss_start_matches_paper_operator_form(ciss_angle_rad):
    # Luo & Hore eq. (10), written with spin operators instead of kets: an independent route to the same state.
    # P = E/4 − S1z S2z − (S1x S2x + S1y S2y) cos χ + ½ (S1z − S2z) sin χ, electron 1 = D, electron 2 = A.
    sx, sy, sz = spin_matrices(0.5)
    one = np.eye(2)
    s1 = [np.kron(matrix, one) for matrix in (sx, sy, sz)]
    s2 = [np.kron(one, matrix) for matrix in (sx, sy, sz)]
    paper = (np.eye(4) / 4 - s1[2] @ s2[2] - (s1[0] @ s2[0] + s1[1] @ s2[1]) * np.cos(ciss_angle_rad)
             + 0.5 * (s1[2] - s2[2]) * np.sin(ciss_angle_rad))
    np.testing.assert_allclose(ciss_density_matrix(ciss_angle_rad), paper, atol=1e-14)


def test_singlet_and_triplet_fill_the_electron_space():
    # |S⟩⟨S| + 3 × (triplet mixture) = identity, and the two states do not overlap.
    singlet = singlet_density_matrix()
    triplet = triplet_density_matrix()
    np.testing.assert_allclose(singlet + 3 * triplet, np.eye(4), atol=1e-15)
    assert np.isclose(np.trace(singlet @ triplet), 0.0)


# %% Hypothesis, variants and presets

def test_variant_changes_only_the_copy():
    original = baseline_hypothesis(0.0)
    triplet_born = original.variant(name="triplet-born", electron_initial_state=triplet_density_matrix())
    assert triplet_born.name == "triplet-born"
    np.testing.assert_array_equal(triplet_born.electron_initial_state, triplet_density_matrix())
    np.testing.assert_array_equal(original.electron_initial_state, singlet_density_matrix())
    assert triplet_born.nuclei is original.nuclei  # unchanged fields are carried over


def test_hypothesis_fields_cannot_be_reassigned():
    hypothesis = baseline_hypothesis(0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        hypothesis.singlet_recombination_rate_per_us = 2.0


def test_baseline_preset_has_old_parameters():
    # Expected values come from DISCO (old repo, All DISCO/DISCO.py lines 149-152) and may rely on
    # stale placeholder parameters. Here they only pin the preset to the old model.
    hypothesis = baseline_hypothesis(0.3)
    assert len(hypothesis.nuclei) == 1
    nucleus = hypothesis.nuclei[0]
    assert nucleus.radical == 1 and nucleus.spin == 1
    np.testing.assert_allclose(np.diag(nucleus.hyperfine_tensor_MHz), [28 * 0.7, 28 * 0.7, 28 * 1.6])
    assert hypothesis.exchange_coupling_MHz == 1.0
    assert hypothesis.zeeman_frequency_MHz == 28.0
    assert hypothesis.singlet_recombination_rate_per_us == 1.0
    assert hypothesis.triplet_recombination_rate_per_us == 1.0
    np.testing.assert_array_equal(hypothesis.electron_initial_state, singlet_density_matrix())


# %% Spin operators and static Hamiltonian

def hydrogen(radical, hyperfine_coupling_MHz):
    """A spin-1/2 nucleus with an isotropic hyperfine coupling, for tests."""
    return Nucleus(radical=radical, spin=0.5, hyperfine_tensor_MHz=hyperfine_coupling_MHz * np.eye(3))


def test_full_space_dimension_follows_the_nuclei():
    hypothesis = baseline_hypothesis(0.0)
    assert spin_operators(hypothesis)[0][0].shape == (12, 12)  # 2 × 2 × 3
    assert spin_operators(hypothesis.variant(nuclei=()))[0][0].shape == (4, 4)
    two_nuclei = hypothesis.variant(nuclei=hypothesis.nuclei + (hydrogen(2, 5.0),))
    assert spin_operators(two_nuclei)[0][0].shape == (24, 24)  # 2 × 2 × 3 × 2


def test_spin_operators_obey_spin_algebra():
    # For every spin: [Sx, Sy] = i Sz and Sx² + Sy² + Sz² = s(s+1). Different spins commute.
    baseline = baseline_hypothesis(0.0)
    operators = spin_operators(baseline.variant(nuclei=baseline.nuclei + (hydrogen(2, 5.0),)))
    spins = [0.5, 0.5, 1.0, 0.5]
    for (x, y, z), spin in zip(operators, spins):
        np.testing.assert_allclose(x @ y - y @ x, 1j * z, atol=1e-12)
        np.testing.assert_allclose(x @ x + y @ y + z @ z, spin * (spin + 1) * np.eye(24), atol=1e-12)
    electron_1_x, nucleus_y = operators[0][0], operators[2][1]
    np.testing.assert_allclose(electron_1_x @ nucleus_y, nucleus_y @ electron_1_x, atol=1e-12)


@pytest.mark.parametrize("hyperfine_coupling_MHz, delta, exchange_coupling_MHz, zeeman_frequency_MHz", [
    (28.0, 0.0, 1.0, 28.0),   # DISCO values (old repo); may rely on stale placeholder parameters
    (28.0, 0.3, 1.0, 28.0),   # DISCO values (old repo); may rely on stale placeholder parameters
    (10.0, 0.2, -3.0, 7.0),   # arbitrary values, so no term can hide behind a special number
])
def test_hamiltonian_matches_reference_engine(hyperfine_coupling_MHz, delta, exchange_coupling_MHz,
                                              zeeman_frequency_MHz):
    nucleus = Nucleus(radical=1, spin=1,
                      hyperfine_tensor_MHz=axial_hyperfine_tensor_MHz(hyperfine_coupling_MHz, delta))
    hypothesis = baseline_hypothesis(delta).variant(nuclei=(nucleus,),
                                                    exchange_coupling_MHz=exchange_coupling_MHz,
                                                    zeeman_frequency_MHz=zeeman_frequency_MHz)
    reference = reference_engine.static_hamiltonian(reference_engine.spin_operators(), hyperfine_coupling_MHz,
                                                    delta, exchange_coupling_MHz, zeeman_frequency_MHz).full()
    hamiltonian = static_hamiltonian(hypothesis)
    np.testing.assert_allclose(hamiltonian, reference, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)  # Hermitian


def test_isotropic_hydrogen_on_radical_2_gives_known_energies():
    # H = A S2·I only (J = 0, ω0 = 0). S2·I is +1/4 on the three triplet states of (S2, I) and
    # −3/4 on the singlet; electron 1 doubles every level. Checks radical 2 and a spin-1/2 nucleus.
    hyperfine_coupling_MHz = 5.0
    hypothesis = baseline_hypothesis(0.0).variant(nuclei=(hydrogen(2, hyperfine_coupling_MHz),),
                                                  exchange_coupling_MHz=0.0, zeeman_frequency_MHz=0.0)
    energies = np.linalg.eigvalsh(static_hamiltonian(hypothesis))
    expected = angular_frequency(hyperfine_coupling_MHz) * np.array([-0.75] * 2 + [0.25] * 6)
    np.testing.assert_allclose(energies, expected, atol=1e-12)


def test_nucleus_on_a_missing_radical_is_rejected():
    hypothesis = baseline_hypothesis(0.0).variant(nuclei=(hydrogen(3, 5.0),))
    with pytest.raises(ValueError):
        static_hamiltonian(hypothesis)


# %% Initial state and Haberkorn recombination

def test_singlet_projector_matches_reference_engine():
    projector = singlet_projector(baseline_hypothesis(0.0))
    np.testing.assert_allclose(projector, reference_engine.singlet_projector().full(), atol=1e-15)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-15)  # a projector
    assert np.isclose(np.trace(projector), 3.0)  # one singlet for each of the 3 nuclear states


def test_initial_state_equals_old_average_over_nuclear_states():
    # The old code started from |S⟩ ⊗ |m⟩ for each nuclear state m and averaged the results.
    # Yields are linear in ρ, so that is the same as starting from the average of |S,m⟩⟨S,m|.
    old_average = sum(qt.ket2dm(qt.tensor(electron_singlet(), qt.basis(3, m))).full() for m in range(3)) / 3
    np.testing.assert_allclose(initial_density_matrix(baseline_hypothesis(0.0)), old_average, atol=1e-15)


@pytest.mark.parametrize("electron_state, expected_singlet_fraction", [
    (singlet_density_matrix(), 1.0),
    (triplet_density_matrix(), 0.0),
])
def test_initial_state_is_valid_and_has_right_singlet_fraction(electron_state, expected_singlet_fraction):
    hypothesis = baseline_hypothesis(0.0).variant(electron_initial_state=electron_state)
    density_matrix = initial_density_matrix(hypothesis)
    assert np.isclose(np.trace(density_matrix), 1.0)
    np.testing.assert_allclose(density_matrix, density_matrix.conj().T, atol=1e-15)
    singlet_fraction = np.trace(singlet_projector(hypothesis) @ density_matrix).real
    assert np.isclose(singlet_fraction, expected_singlet_fraction)


def test_equal_rates_give_K_proportional_to_identity():
    # This is why k_S = k_T reduces to the old model: K = k × identity just multiplies ρ by exp(−kt).
    hypothesis = baseline_hypothesis(0.0).variant(singlet_recombination_rate_per_us=0.7,
                                                  triplet_recombination_rate_per_us=0.7)
    np.testing.assert_allclose(recombination_operator(hypothesis), 0.7 * np.eye(12), atol=1e-15)


def test_unequal_rates_give_K_eigenvalues_k_S_and_k_T():
    # k_S on the 3 singlet states (one per nuclear state), k_T on the 9 triplet states. Rates in µs⁻¹, no 2π.
    hypothesis = baseline_hypothesis(0.0).variant(singlet_recombination_rate_per_us=2.0,
                                                  triplet_recombination_rate_per_us=0.5)
    recombination = recombination_operator(hypothesis)
    np.testing.assert_allclose(np.linalg.eigvalsh(recombination), [0.5] * 9 + [2.0] * 3, atol=1e-12)
    projector = singlet_projector(hypothesis)
    np.testing.assert_allclose(recombination @ projector, 2.0 * projector, atol=1e-12)


def test_effective_hamiltonian_splits_into_H_and_minus_K_over_2():
    # Hermitian part of H − iK/2 is H; anti-Hermitian part is −K/2.
    hypothesis = baseline_hypothesis(0.3).variant(singlet_recombination_rate_per_us=2.0,
                                                  triplet_recombination_rate_per_us=0.5)
    effective = effective_hamiltonian(hypothesis)
    hermitian_part = (effective + effective.conj().T) / 2
    anti_hermitian_part = (effective - effective.conj().T) / 2j
    np.testing.assert_allclose(hermitian_part, static_hamiltonian(hypothesis), atol=1e-12)
    np.testing.assert_allclose(anti_hermitian_part, -recombination_operator(hypothesis) / 2, atol=1e-12)


# %% Static Haberkorn yields (no drive), by a slow test-only propagator

def static_haberkorn_yields(hypothesis, times_us):
    """Singlet and triplet yields with no drive, and the population left at the last time (all dimensionless).

    Steps ρ → U ρ U† with U = expm(−i (H − iK/2) dt), which is exact for a time-independent H.
    Y_S = k_S ∫ Tr(P_S ρ) dt and Y_T = k_T ∫ Tr(P_T ρ) dt by the trapezoid rule. times_us must be evenly spaced.
    """
    step_us = times_us[1] - times_us[0]
    propagator = scipy.linalg.expm(-1j * effective_hamiltonian(hypothesis) * step_us)
    singlet = singlet_projector(hypothesis)
    triplet = np.eye(len(singlet)) - singlet
    density_matrix = initial_density_matrix(hypothesis)
    singlet_population = np.zeros(len(times_us))
    triplet_population = np.zeros(len(times_us))
    for i in range(len(times_us)):
        singlet_population[i] = np.trace(singlet @ density_matrix).real
        triplet_population[i] = np.trace(triplet @ density_matrix).real
        density_matrix = propagator @ density_matrix @ propagator.conj().T
    singlet_yield = hypothesis.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us)
    triplet_yield = hypothesis.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us)
    remaining = singlet_population[-1] + triplet_population[-1]
    return singlet_yield, triplet_yield, remaining


def frozen_spins(singlet_rate_per_us, triplet_rate_per_us, electron_initial_state):
    """A pair with no nuclei, J = 0 and ω0 = 0, so H = 0 and only recombination acts."""
    return baseline_hypothesis(0.0).variant(nuclei=(), exchange_coupling_MHz=0.0, zeeman_frequency_MHz=0.0,
                                            electron_initial_state=electron_initial_state,
                                            singlet_recombination_rate_per_us=singlet_rate_per_us,
                                            triplet_recombination_rate_per_us=triplet_rate_per_us)


def test_frozen_singlet_decays_at_k_S_only():
    # Singlet population is exp(−k_S t), so Y_S = 1 − exp(−k_S T) and Y_T = 0.
    # Trapezoid error on this grid is about 3e-6 (measured), hence the tolerance.
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, triplet_yield, _ = static_haberkorn_yields(frozen_spins(2.0, 0.5, singlet_density_matrix()),
                                                              times_us)
    assert abs(singlet_yield - (1 - np.exp(-2.0 * 6))) < 1e-5
    assert abs(triplet_yield) < 1e-12


def test_frozen_triplet_never_gives_singlet_yield():
    # Triplet population is exp(−k_T t), so Y_T = 1 − exp(−k_T T) and Y_S = 0.
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, triplet_yield, _ = static_haberkorn_yields(frozen_spins(2.0, 0.5, triplet_density_matrix()),
                                                              times_us)
    assert abs(singlet_yield) < 1e-12
    assert abs(triplet_yield - (1 - np.exp(-0.5 * 6))) < 1e-5


def test_probability_is_conserved_with_unequal_rates():
    # d(Tr ρ)/dt = −Tr(K ρ), so Y_S + Y_T + Tr ρ(T) = 1 up to trapezoid error.
    # Measured error: 3e-6 on 2000 points, 3e-8 on 20000 points (it falls as step²), so
    # a fine grid tests the physics rather than the integration rule.
    times_us = np.linspace(0, 6, 20000)
    hypothesis = baseline_hypothesis(0.3).variant(singlet_recombination_rate_per_us=2.0,
                                                  triplet_recombination_rate_per_us=0.5)
    singlet_yield, triplet_yield, remaining = static_haberkorn_yields(hypothesis, times_us)
    assert abs(singlet_yield + triplet_yield + remaining - 1) < 1e-6


@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_equal_rates_reproduce_reference_engine_static_yield(delta):
    # With k_S = k_T the Haberkorn model must give the old yield ∫ k exp(−kt) P_S dt.
    # Baseline parameters and the 0–6 µs, 2000-point grid come from DISCO (old repo) and may rely on
    # stale placeholder parameters. Measured agreement is about 2e-9.
    times_us = np.linspace(0, 6, 2000)
    singlet_yield, _, _ = static_haberkorn_yields(baseline_hypothesis(delta), times_us)
    operators = reference_engine.spin_operators()
    reference_hamiltonian = reference_engine.static_hamiltonian(operators, 28.0, delta, 1.0, 28.0)
    reference_population = reference_engine.singlet_population(reference_hamiltonian, times_us)
    reference_yield = reference_engine.singlet_yield(reference_population, times_us, 1.0)
    assert abs(singlet_yield - reference_yield) < 1e-6


# %% Orientation: turning the molecule relative to the field

# Directions (polar angle θ, azimuth φ in radians) used below: along the field, tilted, and perpendicular.
DIRECTIONS = [(0.0, 0.0), (0.7, 0.0), (1.0, 2.0), (np.pi / 2, 0.3)]


def ciss_test_pair():
    """Baseline pair (anisotropy 0.3), CISS-born at χ = 0.25π. Baseline parameters come from DISCO (old repo) and
    may rely on stale placeholder parameters."""
    return baseline_hypothesis(0.3).variant(electron_initial_state=ciss_density_matrix(0.25 * np.pi))


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_rotation_turns_z_to_the_given_direction(polar_angle_rad, azimuth_rad):
    rotation = rotation_matrix(polar_angle_rad, azimuth_rad)
    direction = [np.sin(polar_angle_rad) * np.cos(azimuth_rad), np.sin(polar_angle_rad) * np.sin(azimuth_rad),
                 np.cos(polar_angle_rad)]
    assert np.allclose(rotation @ [0, 0, 1], direction, atol=1e-15)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-15)  # a rotation: lengths kept
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-15)  # and no mirror


def test_no_turn_changes_nothing():
    hypothesis = ciss_test_pair()
    turned = oriented_hypothesis(hypothesis, 0.0, 0.0)
    assert np.allclose(turned.nuclei[0].hyperfine_tensor_MHz, hypothesis.nuclei[0].hyperfine_tensor_MHz, atol=1e-15)
    assert np.allclose(turned.electron_initial_state, hypothesis.electron_initial_state, atol=1e-15)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_hyperfine_axis_follows_the_molecule(polar_angle_rad, azimuth_rad):
    # The axial tensor's unique axis (eigenvalue A(1 + 2δ)) must point along the molecule's new z axis.
    turned = oriented_hypothesis(baseline_hypothesis(0.3), polar_angle_rad, azimuth_rad)
    eigenvalues, eigenvectors = np.linalg.eigh(turned.nuclei[0].hyperfine_tensor_MHz)
    assert eigenvalues[-1] == pytest.approx(28.0 * 1.6)  # DISCO A = 28 MHz (may be stale), δ = 0.3
    assert abs(eigenvectors[:, -1] @ rotation_matrix(polar_angle_rad, azimuth_rad)[:, 2]) == pytest.approx(1.0)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_ciss_spins_point_along_the_molecule_axis(polar_angle_rad, azimuth_rad):
    # χ = π/2 is electron 1 up and electron 2 down along the CISS axis. After turning, the average spins must be
    # +½ and −½ times the molecule's new z axis. This checks that the start state turns the same way as the tensor.
    turned = oriented_hypothesis(baseline_hypothesis(0.3).variant(electron_initial_state=ciss_density_matrix(np.pi / 2)),
                                 polar_angle_rad, azimuth_rad)
    axis = rotation_matrix(polar_angle_rad, azimuth_rad)[:, 2]
    one_spin = spin_matrices(0.5)
    electron_1 = [np.kron(matrix, np.eye(2)) for matrix in one_spin]
    electron_2 = [np.kron(np.eye(2), matrix) for matrix in one_spin]
    state = turned.electron_initial_state
    assert np.allclose([np.trace(operator @ state).real for operator in electron_1], 0.5 * axis, atol=1e-14)
    assert np.allclose([np.trace(operator @ state).real for operator in electron_2], -0.5 * axis, atol=1e-14)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_singlet_start_is_unchanged_by_turning(polar_angle_rad, azimuth_rad):
    # The singlet has total spin zero, so it has no direction to turn.
    turned = oriented_hypothesis(baseline_hypothesis(0.3), polar_angle_rad, azimuth_rad)
    assert np.allclose(turned.electron_initial_state, singlet_density_matrix(), atol=1e-15)


def test_without_a_field_orientation_cannot_matter():
    # With no field and no drive nothing picks out a lab direction, so turning the whole molecule (tensor and start
    # state together) cannot change the yields. Measured difference: 3e-15.
    times_us = np.linspace(0, 6, 2000)
    no_field = ciss_test_pair().variant(zeeman_frequency_MHz=0.0)
    aligned = static_haberkorn_yields(no_field, times_us)
    turned = static_haberkorn_yields(oriented_hypothesis(no_field, 1.0, 2.0), times_us)
    assert np.allclose(aligned, turned, atol=1e-12)


def test_spinning_about_the_field_changes_nothing_without_a_drive():
    # The field lies along z, so with no drive, turning the molecule about z (changing φ only) is a symmetry.
    # Measured difference: 7e-16. Tilting (changing θ) does change the yield, by about 0.06 here.
    times_us = np.linspace(0, 6, 2000)
    first = static_haberkorn_yields(oriented_hypothesis(ciss_test_pair(), 0.7, 0.0), times_us)
    second = static_haberkorn_yields(oriented_hypothesis(ciss_test_pair(), 0.7, 1.9), times_us)
    aligned = static_haberkorn_yields(ciss_test_pair(), times_us)
    assert np.allclose(first, second, atol=1e-12)
    assert abs(first[0] - aligned[0]) > 0.01


# %% Flavin-tryptophan pair (Daniel's tensors): dipolar coupling, recombination state, field directions

def flavin_tryptophan_test_pairs():
    """Daniel's classical pair (χ = 0, k_r = k_f = 1 µs⁻¹) and CISS pair (χ = π/2, k_r = 10, k_f = 1), at 50 µT."""
    return (flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, 50.0),
            flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, 50.0))


def test_flavin_tryptophan_tensors_match_daniels_numbers():
    # Numbers from the handover note (scratch check of Daniel's tensors, 2026-10-15), recomputed here.
    hypothesis, _ = flavin_tryptophan_test_pairs()
    flavin, tryptophan = hypothesis.nuclei
    assert (flavin.radical, tryptophan.radical, flavin.spin, tryptophan.spin) == (1, 2, 1, 1)
    assert np.trace(angular_frequency(flavin.hyperfine_tensor_MHz)) == pytest.approx(276.43, abs=0.01)  # rad/µs
    assert np.trace(flavin.hyperfine_tensor_MHz) / 3 == pytest.approx(14.665, abs=1e-3)
    np.testing.assert_allclose(np.linalg.eigvalsh(flavin.hyperfine_tensor_MHz), [-2.81, -2.43, 49.24], atol=0.01)
    assert np.trace(tryptophan.hyperfine_tensor_MHz) / 3 == pytest.approx(9.011, abs=1e-3)
    np.testing.assert_allclose(np.linalg.eigvalsh(tryptophan.hyperfine_tensor_MHz), [-1.78, -1.49, 30.30], atol=0.01)
    np.testing.assert_allclose(np.diag(hypothesis.dipolar_tensor_MHz), [9.55, 9.55, -19.11], atol=0.01)
    assert hypothesis.exchange_coupling_MHz == 0.0
    assert len(static_hamiltonian(hypothesis)) == 36


def test_zeeman_frequency_at_geomagnetic_field():
    # About 1.40 MHz at 50 µT (handover scratch check); proportional to the field.
    assert zeeman_frequency_MHz(50.0) == pytest.approx(1.40, abs=0.005)
    assert zeeman_frequency_MHz(100.0) == pytest.approx(2 * zeeman_frequency_MHz(50.0), rel=1e-14)


def test_isotropic_dipolar_tensor_acts_like_exchange():
    # D = d × identity gives d S1·S2, which is the exchange term J(S1·S2 − 1/4) with J = d, up to the constant d/4.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    coupling_MHz = 3.0
    exchange = baseline_hypothesis(0.3).variant(exchange_coupling_MHz=coupling_MHz)
    dipolar = baseline_hypothesis(0.3).variant(exchange_coupling_MHz=0.0, dipolar_tensor_MHz=coupling_MHz * np.eye(3))
    difference = static_hamiltonian(dipolar) - static_hamiltonian(exchange)
    np.testing.assert_allclose(difference, angular_frequency(coupling_MHz) / 4 * np.eye(12), atol=1e-12)


def test_dipolar_off_diagonal_element_couples_the_right_spin_components():
    # A tensor with only D_xy = 2 MHz must give 2π × 2 × S1x S2y, built here independently with QuTiP.
    tensor = np.zeros((3, 3))
    tensor[0, 1] = 2.0
    hypothesis = baseline_hypothesis(0.3).variant(exchange_coupling_MHz=0.0, dipolar_tensor_MHz=tensor)
    without = hypothesis.variant(dipolar_tensor_MHz=np.zeros((3, 3)))
    expected = angular_frequency(2.0) * qt.tensor(qt.jmat(0.5, "x"), qt.jmat(0.5, "y"), qt.qeye(3)).full()
    np.testing.assert_allclose(static_hamiltonian(hypothesis) - static_hamiltonian(without), expected, atol=1e-12)


def test_recombination_state_defaults_to_the_singlet():
    hypothesis = baseline_hypothesis(0.3)
    np.testing.assert_array_equal(hypothesis.recombination_state, singlet_density_matrix())
    np.testing.assert_array_equal(recombination_projector(hypothesis), singlet_projector(hypothesis))


def test_classical_flavin_tryptophan_pair_is_singlet_born_and_recombines_through_the_singlet():
    classical, _ = flavin_tryptophan_test_pairs()
    np.testing.assert_allclose(classical.electron_initial_state, singlet_density_matrix(), atol=1e-15)
    np.testing.assert_allclose(classical.recombination_state, singlet_density_matrix(), atol=1e-15)


def test_ciss_recombination_state_is_the_start_with_minus_chi():
    # Daniel: recombination_projector_ciss(chi) = initial_state_ciss(-chi), i.e. cos(χ/2)|S⟩ − sin(χ/2)|T0⟩.
    chi = 0.3 * np.pi
    hypothesis = flavin_tryptophan_hypothesis(chi, 10.0, 1.0, 50.0)
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    triplet_zero = np.array([0, 1, 1, 0]) / np.sqrt(2)
    ket = np.cos(chi / 2) * singlet - np.sin(chi / 2) * triplet_zero
    np.testing.assert_allclose(hypothesis.recombination_state, np.outer(ket, ket), atol=1e-15)


def test_recombination_operator_is_k_r_P_R_plus_k_f_identity():
    _, ciss = flavin_tryptophan_test_pairs()  # k_r = 10, k_f = 1
    expected = 10.0 * recombination_projector(ciss) + 1.0 * np.eye(36)
    np.testing.assert_allclose(recombination_operator(ciss), expected, atol=1e-14)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_field_direction_sees_the_tensor_along_that_direction(polar_angle_rad, azimuth_rad):
    # With the field along n (molecule frame), the turned tensor's zz element (along the lab field) must be nᵀAn,
    # and its xx element (along the lab drive) e_θᵀ A e_θ, e_θ = (cos θ cos φ, cos θ sin φ, −sin θ).
    hypothesis, _ = flavin_tryptophan_test_pairs()
    turned = field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
    tensor = hypothesis.nuclei[0].hyperfine_tensor_MHz
    n = np.array([np.sin(polar_angle_rad) * np.cos(azimuth_rad), np.sin(polar_angle_rad) * np.sin(azimuth_rad),
                  np.cos(polar_angle_rad)])
    e_theta = np.array([np.cos(polar_angle_rad) * np.cos(azimuth_rad), np.cos(polar_angle_rad) * np.sin(azimuth_rad),
                        -np.sin(polar_angle_rad)])
    assert turned.nuclei[0].hyperfine_tensor_MHz[2, 2] == pytest.approx(n @ tensor @ n, abs=1e-12)
    assert turned.nuclei[0].hyperfine_tensor_MHz[0, 0] == pytest.approx(e_theta @ tensor @ e_theta, abs=1e-12)
    assert turned.dipolar_tensor_MHz[2, 2] == pytest.approx(n @ hypothesis.dipolar_tensor_MHz @ n, abs=1e-12)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_field_direction_turn_undoes_orienting_the_molecule(polar_angle_rad, azimuth_rad):
    # oriented_hypothesis turns the molecule by R; field_direction_hypothesis turns it back by the reverse of R.
    _, hypothesis = flavin_tryptophan_test_pairs()
    back = field_direction_hypothesis(oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad),
                                      polar_angle_rad, azimuth_rad)
    for original, returned in zip(hypothesis.nuclei, back.nuclei):
        np.testing.assert_allclose(returned.hyperfine_tensor_MHz, original.hyperfine_tensor_MHz, atol=1e-12)
    np.testing.assert_allclose(back.dipolar_tensor_MHz, hypothesis.dipolar_tensor_MHz, atol=1e-12)
    np.testing.assert_allclose(back.electron_initial_state, hypothesis.electron_initial_state, atol=1e-14)
    np.testing.assert_allclose(back.recombination_state, hypothesis.recombination_state, atol=1e-14)


@pytest.mark.parametrize("polar_angle_rad, azimuth_rad", DIRECTIONS)
def test_ciss_spins_and_recombination_state_turn_with_the_tensors(polar_angle_rad, azimuth_rad):
    # χ = π/2: electron 1 up along the molecule's z axis. With the field along n, the molecule's z axis lies along
    # the third column of the turn applied, Rᵀ. The start and recombination states must keep their overlap.
    _, hypothesis = flavin_tryptophan_test_pairs()
    turned = field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad)
    molecule_z_in_lab = rotation_matrix(polar_angle_rad, azimuth_rad).T[:, 2]
    electron_1 = [np.kron(matrix, np.eye(2)) for matrix in spin_matrices(0.5)]
    spins = [np.trace(operator @ turned.electron_initial_state).real for operator in electron_1]
    np.testing.assert_allclose(spins, 0.5 * molecule_z_in_lab, atol=1e-14)
    overlap = np.trace(hypothesis.recombination_state @ hypothesis.electron_initial_state).real
    turned_overlap = np.trace(turned.recombination_state @ turned.electron_initial_state).real
    assert turned_overlap == pytest.approx(overlap, abs=1e-14)


def test_azimuth_changes_the_field_seen_by_a_molecule_with_no_symmetry_axis():
    # Why field_direction_hypothesis exists: oriented_hypothesis's φ does not move the field within the molecule
    # (the field-direction element of the tensor stays the same), field_direction_hypothesis's φ does.
    hypothesis, _ = flavin_tryptophan_test_pairs()
    first = oriented_hypothesis(hypothesis, 0.7, 0.0).nuclei[0].hyperfine_tensor_MHz[2, 2]
    second = oriented_hypothesis(hypothesis, 0.7, 1.9).nuclei[0].hyperfine_tensor_MHz[2, 2]
    assert first == pytest.approx(second, abs=1e-12)
    first = field_direction_hypothesis(hypothesis, 0.7, 0.0).nuclei[0].hyperfine_tensor_MHz[2, 2]
    second = field_direction_hypothesis(hypothesis, 0.7, 1.9).nuclei[0].hyperfine_tensor_MHz[2, 2]
    assert abs(first - second) > 0.1  # measured: 0.45 MHz
