import dataclasses
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
import scipy.constants

# Units: settings in MHz, µs and µs⁻¹; matrices in rad/µs. MHz -> rad/µs happens only in angular_frequency.
# Rates are probabilities per µs and never get the 2π.
# Electron basis order: ↑↑, ↑↓, ↓↑, ↓↓ (electron 1 first).


# new
def angular_frequency(frequency_MHz):
    return 2 * np.pi * frequency_MHz


# new
def zeeman_frequency_MHz(field_strength_uT):
    # free-electron g for both radicals (CODATA value via scipy); 50 µT -> about 1.40 MHz
    gyromagnetic_ratio_MHz_per_T = abs(scipy.constants.physical_constants["electron gyromag. ratio in MHz/T"][0])
    return gyromagnetic_ratio_MHz_per_T * field_strength_uT * 1e-6


# old, edited: numpy instead of QuTiP
def singlet_density_matrix():
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    return np.outer(singlet, singlet).astype(complex)


# new
def no_dipolar_coupling_MHz():
    return np.zeros((3, 3))


# new
@dataclass(frozen=True)
class Nucleus:
    radical: int  # which electron it couples to, 1 or 2
    spin: float
    hyperfine_tensor_MHz: np.ndarray  # 3x3, in the frame the field is applied in


# new
@dataclass(frozen=True)
class Hypothesis:
    name: str
    nuclei: tuple
    exchange_coupling_MHz: float
    zeeman_frequency_MHz: float  # field along lab z
    electron_initial_state: np.ndarray  # 4x4; nuclei always start fully mixed
    singlet_recombination_rate_per_us: float  # k_S, out of the recombination state
    triplet_recombination_rate_per_us: float  # k_T, out of everything else
    recombination_state: np.ndarray = field(default_factory=singlet_density_matrix)
    dipolar_tensor_MHz: np.ndarray = field(default_factory=no_dipolar_coupling_MHz)

    def variant(self, **changes):
        return dataclasses.replace(self, **changes)


# new
def ciss_density_matrix(ciss_angle_rad):
    """CISS electron state cos(χ/2)|S⟩ + sin(χ/2)|T0⟩ (Luo & Hore 2021, eq. 8), quantised along the molecule's z."""
    # χ = 0 is the singlet, χ = π/2 is |↑↓⟩ (electron 1 up). Daniel's recombination state is the same with -χ.
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    triplet_zero = np.array([0, 1, 1, 0]) / np.sqrt(2)
    state = np.cos(ciss_angle_rad / 2) * singlet + np.sin(ciss_angle_rad / 2) * triplet_zero
    return np.outer(state, state).astype(complex)


# old, edited: any spin quantum number
def spin_matrices(spin):
    # the only QuTiP call left in the code
    return [qt.jmat(spin, axis).full() for axis in ("x", "y", "z")]


# old, edited: any number of nuclei on either radical
def spin_operators(hypothesis):
    # order: electron 1, electron 2, then nuclei as listed
    spins = [0.5, 0.5] + [nucleus.spin for nucleus in hypothesis.nuclei]
    dimensions = [int(2 * spin + 1) for spin in spins]
    operators = []
    for position, spin in enumerate(spins):
        identity_before = np.eye(int(np.prod(dimensions[:position])))
        identity_after = np.eye(int(np.prod(dimensions[position + 1:])))
        operators.append([np.kron(np.kron(identity_before, matrix), identity_after)
                          for matrix in spin_matrices(spin)])
    return operators


# old, edited: full tensors, nuclei on both radicals, dipolar term
def static_hamiltonian(hypothesis):
    """H = Σ S·A·I + J(S1·S2 - 1/4) + S1·D·S2 + ω0(S1z + S2z) in rad/µs; field along lab z."""
    # Tensors are taken as given, so the field direction relative to the molecule is set by turning the tensors
    # (field_direction_hypothesis), not by tilting the Zeeman term.
    operators = spin_operators(hypothesis)
    electron_1, electron_2 = operators[0], operators[1]
    identity = np.eye(electron_1[0].shape[0])

    hyperfine_term = np.zeros_like(identity, dtype=complex)
    for nucleus, nuclear_spin in zip(hypothesis.nuclei, operators[2:]):
        if nucleus.radical not in (1, 2):
            raise ValueError(f"Nucleus radical must be 1 or 2, got {nucleus.radical}")
        electron = operators[nucleus.radical - 1]
        tensor = angular_frequency(nucleus.hyperfine_tensor_MHz)
        for a in range(3):
            for b in range(3):
                hyperfine_term += tensor[a, b] * electron[a] @ nuclear_spin[b]

    # for the flavin-tryptophan pair J = 0, so the dipolar term is the only coupling between the radicals
    spin_dot_product = sum(electron_1[a] @ electron_2[a] for a in range(3))
    exchange_term = angular_frequency(hypothesis.exchange_coupling_MHz) * (spin_dot_product - 0.25 * identity)
    dipolar = angular_frequency(hypothesis.dipolar_tensor_MHz)
    dipolar_term = sum(dipolar[a, b] * electron_1[a] @ electron_2[b] for a in range(3) for b in range(3))
    zeeman_term = angular_frequency(hypothesis.zeeman_frequency_MHz) * (electron_1[2] + electron_2[2])
    return hyperfine_term + exchange_term + dipolar_term + zeeman_term


# old
def drive_operator(hypothesis):
    # drive along lab x on both electrons, perpendicular to the static field
    operators = spin_operators(hypothesis)
    return operators[0][0] + operators[1][0]


# new
def nuclear_dimension(hypothesis):
    return int(np.prod([2 * nucleus.spin + 1 for nucleus in hypothesis.nuclei]))


# old, edited: any recombination state, not only the singlet
def recombination_projector(hypothesis):
    # P_R = |recombination state⟩⟨...| ⊗ 1 on the nuclei
    return np.kron(hypothesis.recombination_state, np.eye(nuclear_dimension(hypothesis)))


# new
def recombination_operator(hypothesis):
    """Haberkorn K = k_S P_R + k_T (1 - P_R), in µs⁻¹."""
    # with k_S = k_r + k_f and k_T = k_f this is K = k_r P_R + k_f 1 (see flavin_tryptophan_hypothesis)
    projector = recombination_projector(hypothesis)
    rest = np.eye(projector.shape[0]) - projector
    return (hypothesis.singlet_recombination_rate_per_us * projector
            + hypothesis.triplet_recombination_rate_per_us * rest)


# new
def effective_hamiltonian(hypothesis):
    # K enters here: H - iK/2 is exact for Haberkorn kinetics, so kets can be evolved instead of ρ
    return static_hamiltonian(hypothesis) - 0.5j * recombination_operator(hypothesis)


# new
def rotation_matrix(polar_angle_rad, azimuth_rad):
    # R = Rz(φ) Ry(θ), takes z to (sin θ cos φ, sin θ sin φ, cos θ)
    tilt = np.array([[np.cos(polar_angle_rad), 0, np.sin(polar_angle_rad)],
                     [0, 1, 0],
                     [-np.sin(polar_angle_rad), 0, np.cos(polar_angle_rad)]])
    turn = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                     [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
                     [0, 0, 1]])
    return turn @ tilt


# new
def spin_half_rotation(polar_angle_rad, azimuth_rad):
    # same rotation on one spin-1/2: exp(-iφSz) exp(-iθSy)
    half_tilt, half_turn = polar_angle_rad / 2, azimuth_rad / 2
    spin_tilt = np.array([[np.cos(half_tilt), -np.sin(half_tilt)],
                          [np.sin(half_tilt), np.cos(half_tilt)]])
    spin_turn = np.diag([np.exp(-1j * half_turn), np.exp(1j * half_turn)])
    return spin_turn @ spin_tilt


# new
def turned_hypothesis(hypothesis, rotation, spin_rotation, name):
    # Turns the whole molecule: hyperfine tensors and the dipolar tensor as R T Rᵀ, the electron start state and the
    # recombination state as U ρ U† with U the same rotation on both electrons. The singlet is unchanged by this;
    # the CISS state and its recombination projector are not, so they turn with the molecule. Field (lab z), drive
    # (lab x) and exchange stay put; the nuclei start fully mixed, so there is no nuclear state to turn.
    nuclei = tuple(dataclasses.replace(nucleus,
                                       hyperfine_tensor_MHz=rotation @ nucleus.hyperfine_tensor_MHz @ rotation.T)
                   for nucleus in hypothesis.nuclei)
    both_electrons = np.kron(spin_rotation, spin_rotation)
    return hypothesis.variant(
        name=name, nuclei=nuclei, dipolar_tensor_MHz=rotation @ hypothesis.dipolar_tensor_MHz @ rotation.T,
        electron_initial_state=both_electrons @ hypothesis.electron_initial_state @ both_electrons.conj().T,
        recombination_state=both_electrons @ hypothesis.recombination_state @ both_electrons.conj().T)


# new
def oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad):
    # molecule's z axis to (θ, φ) in the lab; used only for orientation averaging, which is off for these runs
    # check: for a molecule with no symmetry axis the field then only ever lies in the molecule's xz plane
    return turned_hypothesis(hypothesis, rotation_matrix(polar_angle_rad, azimuth_rad),
                             spin_half_rotation(polar_angle_rad, azimuth_rad),
                             f"{hypothesis.name}, axis at theta={polar_angle_rad:.4g}, phi={azimuth_rad:.4g} rad")


# new
def field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad):
    """Field along n = (sin θ cos φ, sin θ sin φ, cos θ) of the molecule's frame, by turning the molecule with Rᵀ."""
    # Rᵀ brings n onto lab z. Scans use θ in [0, π], φ in [-π, π). Checked numerically: the result equals a full
    # spin rotation of "molecule fixed, field along n" (tensors, CISS state, P_R all consistent, to ~1e-14).
    # The drive (lab x) then lies along (cos θ cos φ, cos θ sin φ, -sin θ) in the molecule's frame.
    # check: that drive direction about the field is a convention; it matters for a molecule with no symmetry axis
    return turned_hypothesis(hypothesis, rotation_matrix(polar_angle_rad, azimuth_rad).T,
                             spin_half_rotation(polar_angle_rad, azimuth_rad).conj().T,
                             f"{hypothesis.name}, field at theta={polar_angle_rad:.4g}, phi={azimuth_rad:.4g} rad")


# new
def flavin_tryptophan_hypothesis(ciss_angle_rad, recombination_rate_per_us, forward_rate_per_us, field_strength_uT):
    """FN5 on electron 1, WN1 on electron 2, dipolar EED, no exchange; 2 x 2 x 3 x 3 = 36 states."""
    # Tensors from Daniel, in rad/µs in the molecule's frame; divided by 2π once below to store MHz.
    FN5_rad_per_us = np.array([[79.48092385, 137.59553919, 56.68450732],
                               [137.59553919, 181.20322632, 79.23527745],
                               [56.68450732, 79.23527745, 15.748773]])
    WN1_rad_per_us = np.array([[51.01430479, 16.46786544, 90.57082545],
                               [16.46786544, -6.75181867, 23.32643811],
                               [90.57082545, 23.32643811, 125.59269667]])
    EED_rad_per_us = np.diag([6.00327982e+01, 6.00327982e+01, -1.20065596e+02])  # axial about molecule z
    exchange_coupling_MHz = 0.0  # Daniel: neglect exchange for separated radicals

    nuclei = (Nucleus(radical=1, spin=1, hyperfine_tensor_MHz=FN5_rad_per_us / (2 * np.pi)),
              Nucleus(radical=2, spin=1, hyperfine_tensor_MHz=WN1_rad_per_us / (2 * np.pi)))
    # Rates read as K = k_r P_R + k_f 1: recombination at k_r through P_R, forward reaction at k_f from any state.
    # check: that reading of Daniel's k_r, k_f is mine; CISS axis = molecule z (EED axis), electron 1 the donor
    return Hypothesis(
        name=f"flavin-tryptophan, chi={ciss_angle_rad:.4g} rad, k_r={recombination_rate_per_us}, "
             f"k_f={forward_rate_per_us} per us, {field_strength_uT} uT",
        nuclei=nuclei,
        exchange_coupling_MHz=exchange_coupling_MHz,
        zeeman_frequency_MHz=zeeman_frequency_MHz(field_strength_uT),
        electron_initial_state=ciss_density_matrix(ciss_angle_rad),
        singlet_recombination_rate_per_us=recombination_rate_per_us + forward_rate_per_us,
        triplet_recombination_rate_per_us=forward_rate_per_us,
        recombination_state=ciss_density_matrix(-ciss_angle_rad),
        dipolar_tensor_MHz=EED_rad_per_us / (2 * np.pi),
    )
