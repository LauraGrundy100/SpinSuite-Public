"""Slow, readable reference engine: the old repo's engine (All DISCO/DISCO.py lines 170-273), tidied.

The physics is unchanged from the old code. Tidying: values are passed as arguments
instead of globals, MHz are converted with model.angular_frequency, and names are
spelled out. The fast engine (engine.py, milestone 3) is tested against this.

Solver tolerances (changed on purpose, 2026-10-06): the old code ran QuTiP sesolve at its
default tolerances, which leaves an error of about 3e-5 in the static singlet yield
(checked against exact diagonalisation). Here sesolve runs at tight tolerances (SOLVER_OPTIONS),
so this engine is accurate enough to check the fast engine to about 1e-6. As a result the
old repo's numbers are matched only within the old engine's own error (about 3e-5 or less),
not exactly.

State space: electron 1 ⊗ electron 2 ⊗ spin-1 nucleus on radical 1, so 2 × 2 × 3 = 12.
Hamiltonians are in rad/µs, times in µs, rates in µs⁻¹.

The last section ("General Haberkorn reference") is new, not from the old repo. It handles any
hypothesis (k_S ≠ k_T, any initial state, any nuclei) by solving the density-matrix equation
literally, so the fast engine can be checked where the old engine cannot go.

"Orientation average, one direction at a time" (milestone 8) is the plain version of
engine.averaged_yields_together, which steps all directions together (batched).
"""

import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp

import engine
import model
from model import angular_frequency

# Tight sesolve tolerances (see module docstring). The old code used QuTiP's defaults.
SOLVER_OPTIONS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 1_000_000}


def spin_operators():
    """Spin operators (dimensionless, ħ = 1) on the 12-dimensional space, as a dict.

    Keys: S1x, S1y, S1z (electron 1), S2x, S2y, S2z (electron 2), Ix, Iy, Iz (nucleus), identity.
    """
    identity_electron = qt.qeye(2)
    identity_nucleus = qt.qeye(3)
    return {
        "S1x": qt.tensor(qt.spin_Jx(0.5), identity_electron, identity_nucleus),
        "S1y": qt.tensor(qt.spin_Jy(0.5), identity_electron, identity_nucleus),
        "S1z": qt.tensor(qt.spin_Jz(0.5), identity_electron, identity_nucleus),
        "S2x": qt.tensor(identity_electron, qt.spin_Jx(0.5), identity_nucleus),
        "S2y": qt.tensor(identity_electron, qt.spin_Jy(0.5), identity_nucleus),
        "S2z": qt.tensor(identity_electron, qt.spin_Jz(0.5), identity_nucleus),
        "Ix": qt.tensor(identity_electron, identity_electron, qt.spin_Jx(1)),
        "Iy": qt.tensor(identity_electron, identity_electron, qt.spin_Jy(1)),
        "Iz": qt.tensor(identity_electron, identity_electron, qt.spin_Jz(1)),
        "identity": qt.tensor(identity_electron, identity_electron, identity_nucleus),
    }


def electron_singlet():
    """Two-electron singlet ket (|↑↓⟩ − |↓↑⟩)/√2."""
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)
    return (qt.tensor(up, down) - qt.tensor(down, up)).unit()


def singlet_projector():
    """Projector onto the electron singlet, |S⟩⟨S| ⊗ identity on the nucleus (12 × 12)."""
    return qt.tensor(qt.ket2dm(electron_singlet()), qt.qeye(3))


def static_hamiltonian(operators, hyperfine_coupling_MHz, delta, exchange_coupling_MHz, zeeman_frequency_MHz):
    """Static Hamiltonian in rad/µs: hyperfine on radical 1, exchange J(S1·S2 − 1/4), Zeeman along z.

    Hyperfine tensor is axial, diag(A(1−δ), A(1−δ), A(1+2δ)), with its unique axis along the field (z).
    delta (δ) is the dimensionless hyperfine anisotropy; A is hyperfine_coupling_MHz.
    """
    op = operators
    hyperfine = angular_frequency(hyperfine_coupling_MHz)
    exchange = angular_frequency(exchange_coupling_MHz)
    zeeman = angular_frequency(zeeman_frequency_MHz)

    hyperfine_term = (hyperfine * (1 - delta) * op["S1x"] * op["Ix"]
                      + hyperfine * (1 - delta) * op["S1y"] * op["Iy"]
                      + hyperfine * (1 + 2 * delta) * op["S1z"] * op["Iz"])
    exchange_term = exchange * (op["S1x"] * op["S2x"] + op["S1y"] * op["S2y"] + op["S1z"] * op["S2z"]
                                - 0.25 * op["identity"])
    zeeman_term = zeeman * (op["S1z"] + op["S2z"])
    return hyperfine_term + exchange_term + zeeman_term


def drive_waveform(times_us, drive_frequencies_MHz, raw_amplitudes, phases_rad):
    """Unscaled drive Σ amp_i · sin(2π f_i t + phase_i) at times in µs (arbitrary units until scaled)."""
    return sum(amplitude * np.sin(angular_frequency(frequency_MHz) * times_us + phase)
               for frequency_MHz, amplitude, phase in zip(drive_frequencies_MHz, raw_amplitudes, phases_rad))


def linf_scale(drive_frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz):
    """Scale factor (rad/µs per unit raw amplitude) making the drive's peak equal the cap (L-infinity).

    As in the old code, the peak is found on 0–12 µs with 20000 points, even though yields use 0–6 µs.
    """
    check_times_us = np.linspace(0, 12, 20000)
    peak = np.max(np.abs(drive_waveform(check_times_us, drive_frequencies_MHz, raw_amplitudes, phases_rad)))
    if peak < 1e-12:  # all-zero drive: nothing to scale (kept from the old code)
        return 1.0
    return angular_frequency(amplitude_cap_MHz) / peak


def drive_coefficient(drive_frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz):
    """Return the function t ↦ scale · waveform(t) in rad/µs, the coefficient of (S1x + S2x) for QuTiP."""
    scale = linf_scale(drive_frequencies_MHz, raw_amplitudes, phases_rad, amplitude_cap_MHz)

    def coefficient(t):
        return scale * drive_waveform(t, drive_frequencies_MHz, raw_amplitudes, phases_rad)

    return coefficient


def singlet_population(hamiltonian, times_us):
    """Singlet population vs time, starting from a singlet, averaged over the three nuclear states.

    hamiltonian is a QuTiP Qobj (static) or a QuTiP list [H_static, [drive_operator, coefficient]].
    """
    projector = singlet_projector()
    population_sum = np.zeros(len(times_us))
    for nuclear_state in range(3):
        initial_ket = qt.tensor(electron_singlet(), qt.basis(3, nuclear_state))
        result = qt.sesolve(hamiltonian, initial_ket, times_us, e_ops=[projector], options=SOLVER_OPTIONS)
        population_sum += result.expect[0]
    return population_sum / 3


def singlet_yield(population, times_us, recombination_rate_per_us):
    """Singlet yield ∫ k·exp(−k t)·P_S(t) dt by the trapezoid rule (dimensionless).

    Equal singlet and triplet recombination rates k, so this is Haberkorn with k_S = k_T.
    """
    weight = recombination_rate_per_us * np.exp(-recombination_rate_per_us * times_us)
    return np.trapezoid(weight * population, times_us)


# %% General Haberkorn reference (new, not from the old repo)

# Tight solve_ivp tolerances, so this reference can check the fast engine to about 1e-6.
DENSITY_MATRIX_SOLVER_OPTIONS = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}


def haberkorn_populations(hypothesis, drive_rad_per_us, times_us):
    """Singlet and triplet populations Tr(P_S ρ), Tr(P_T ρ) at times_us, for any hypothesis. P_S here is the
    recombination projector P_R (model.recombination_projector), P_S itself unless the hypothesis sets another.

    Solves dρ/dt = −i(H_eff ρ − ρ H_eff†), with H_eff = H − iK/2 + f(t)(S1x + S2x), for the full
    density matrix (no kets, no splitting). drive_rad_per_us is f(t) in rad/µs, or None for no drive.
    """
    static_part = model.effective_hamiltonian(hypothesis)
    operators = model.spin_operators(hypothesis)
    drive_operator = operators[0][0] + operators[1][0]
    dimension = len(static_part)

    def rate_of_change(t, flat_density_matrix):
        density_matrix = flat_density_matrix.reshape(dimension, dimension)
        hamiltonian = static_part
        if drive_rad_per_us is not None:
            hamiltonian = static_part + drive_rad_per_us(t) * drive_operator
        return (-1j * (hamiltonian @ density_matrix - density_matrix @ hamiltonian.conj().T)).ravel()

    initial = model.initial_density_matrix(hypothesis).ravel()
    solution = solve_ivp(rate_of_change, (times_us[0], times_us[-1]), initial, t_eval=times_us,
                         **DENSITY_MATRIX_SOLVER_OPTIONS)
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")

    singlet = model.recombination_projector(hypothesis)  # the singlet projector unless the hypothesis sets another
    density_matrices = solution.y.T.reshape(len(times_us), dimension, dimension)
    singlet_population = np.array([np.trace(singlet @ rho).real for rho in density_matrices])
    total_population = np.array([np.trace(rho).real for rho in density_matrices])
    return singlet_population, total_population - singlet_population


def haberkorn_yields(hypothesis, drive_rad_per_us, times_us):
    """Singlet and triplet yields Y_S = k_S ∫ Tr(P_S ρ) dt, Y_T = k_T ∫ Tr(P_T ρ) dt (trapezoid rule), and the population left at the end."""
    singlet_population, triplet_population = haberkorn_populations(hypothesis, drive_rad_per_us, times_us)
    singlet_yield = hypothesis.singlet_recombination_rate_per_us * np.trapezoid(singlet_population, times_us)
    triplet_yield = hypothesis.triplet_recombination_rate_per_us * np.trapezoid(triplet_population, times_us)
    remaining = singlet_population[-1] + triplet_population[-1]
    return singlet_yield, triplet_yield, remaining


# %% Orientation average, one direction at a time (milestone 8)

def averaged_yields_one_direction_at_a_time(hypotheses, drive_rad_per_us, times_us, max_step_us, orientations):
    """Orientation-averaged yields (Y_S, Y_T) per hypothesis: the plain loop that engine.averaged_yields_together
    batches. Each direction in turn: turn every hypothesis there, run engine.yields_together, add with its weight."""
    polar_angles_rad, azimuths_rad, weights = orientations
    averaged = np.zeros((len(hypotheses), 2))
    for polar_angle_rad, azimuth_rad, weight in zip(polar_angles_rad, azimuths_rad, weights):
        oriented = [model.oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad) for hypothesis in hypotheses]
        averaged += weight * np.array(engine.yields_together(oriented, drive_rad_per_us, times_us, max_step_us))
    return [(singlet_yield, triplet_yield) for singlet_yield, triplet_yield in averaged]
