"""Physics definitions: units, and hypotheses (spins, couplings, initial state, recombination).

Units used throughout the repo:
    settings      frequencies and couplings in MHz, times in µs, rates in µs⁻¹
    inside code   frequencies and couplings in rad/µs (angular), times in µs, rates in µs⁻¹

The only conversion is angular_frequency() below. Rates are never multiplied by
2π: a rate is a probability per µs, not an oscillation.

Electron basis order everywhere: ↑↑, ↑↓, ↓↑, ↓↓ (electron 1 first), with ↑ = m_s = +1/2.
"""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import qutip as qt
import scipy.constants


def angular_frequency(frequency_MHz):
    """Convert a frequency or coupling in MHz to an angular frequency in rad/µs (2π × MHz).

    Works on a single number or a NumPy array. Since 1 MHz is one cycle per µs,
    no other factor is needed.
    """
    return 2 * np.pi * frequency_MHz


def zeeman_frequency_MHz(field_strength_uT):
    """Electron Larmor frequency (MHz) in a field of field_strength_uT microtesla, from SciPy's CODATA value of the
    free-electron gyromagnetic ratio (about 28,025 MHz per tesla, so 50 µT gives about 1.40 MHz).

    Assumes g = 2.0023 (free electron) for both radicals; real organic radicals differ by about 0.1%.
    """
    gyromagnetic_ratio_MHz_per_T = abs(scipy.constants.physical_constants["electron gyromag. ratio in MHz/T"][0])
    return gyromagnetic_ratio_MHz_per_T * field_strength_uT * 1e-6


def singlet_density_matrix():
    """Electron singlet |S⟩⟨S|, with |S⟩ = (|↑↓⟩ − |↓↑⟩)/√2, as a 4×4 density matrix."""
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    return np.outer(singlet, singlet).astype(complex)


def no_dipolar_coupling_MHz():
    """A 3×3 tensor of zeros (MHz): the default electron-electron dipolar coupling, i.e. none."""
    return np.zeros((3, 3))


# %% Hypothesis: the physics of one radical pair, as data

@dataclass(frozen=True)
class Nucleus:
    """One magnetic nucleus, hyperfine-coupled to the electron on one radical.

    radical: 1 or 2 (which electron it couples to). spin: nuclear spin quantum number
    (1 for nitrogen-14, 0.5 for hydrogen). hyperfine_tensor_MHz: 3×3 tensor in MHz, lab frame.
    """
    radical: int
    spin: float
    hyperfine_tensor_MHz: np.ndarray


@dataclass(frozen=True)
class Hypothesis:
    """One radical-pair hypothesis: spins, couplings, initial state and recombination rates.

    nuclei: tuple of Nucleus (may be empty). exchange_coupling_MHz: J in J(S1·S2 − 1/4).
    zeeman_frequency_MHz: ω0, same for both electrons, field along z.
    electron_initial_state: 4×4 electron density matrix; the nuclei start fully mixed.
    Recombination is Haberkorn, K = k_S P_R + k_T (1 − P_R), with rates k_S and k_T in µs⁻¹ and P_R the projector
    onto recombination_state (4×4 electron projector, default the singlet, so k_S is the singlet rate as before).
    dipolar_tensor_MHz: 3×3 electron-electron dipolar tensor D in S1·D·S2 (MHz, lab frame; default none).
    Frozen: fields cannot be reassigned. Make changed copies with variant().
    """
    name: str
    nuclei: tuple
    exchange_coupling_MHz: float
    zeeman_frequency_MHz: float
    electron_initial_state: np.ndarray
    singlet_recombination_rate_per_us: float
    triplet_recombination_rate_per_us: float
    recombination_state: np.ndarray = field(default_factory=singlet_density_matrix)
    dipolar_tensor_MHz: np.ndarray = field(default_factory=no_dipolar_coupling_MHz)

    def variant(self, **changes):
        """Return a copy of this hypothesis with the named fields replaced; the original is unchanged."""
        return dataclasses.replace(self, **changes)


def axial_hyperfine_tensor_MHz(hyperfine_coupling_MHz, delta):
    """Axial hyperfine tensor diag(A(1−δ), A(1−δ), A(1+2δ)) in MHz, unique axis along z.

    A is hyperfine_coupling_MHz (the isotropic part, since the trace is 3A for any δ).
    delta (δ) is the dimensionless anisotropy.
    """
    A = hyperfine_coupling_MHz
    return np.diag([A * (1 - delta), A * (1 - delta), A * (1 + 2 * delta)])


def triplet_density_matrix():
    """Equal mixture of the three electron triplets T+, T0, T− (each weight 1/3), as a 4×4 density matrix."""
    triplet_plus = np.array([1, 0, 0, 0])
    triplet_zero = np.array([0, 1, 1, 0]) / np.sqrt(2)
    triplet_minus = np.array([0, 0, 0, 1])
    mixture = sum(np.outer(ket, ket) for ket in (triplet_plus, triplet_zero, triplet_minus)) / 3
    return mixture.astype(complex)


def ciss_density_matrix(ciss_angle_rad):
    """CISS-polarised electron start |ψ⟩⟨ψ|, |ψ⟩ = cos(χ/2)|S⟩ + sin(χ/2)|T0⟩ (Luo & Hore 2021, New J. Phys. 23
    043032, eq. 8), χ = ciss_angle_rad, quantised along z: here the CISS axis lies along the field.

    χ = 0 gives the singlet; χ = π/2 gives |↑↓⟩, 100% CISS (electron 1 up, electron 2 down). Electron 1 plays the
    paper's donor D and electron 2 its acceptor A. Negative χ polarises the other way (opposite chirality).
    By default only the start changes and recombination stays the ordinary singlet kind. The paper's recombination
    state (eq. 11) is the same with −χ: pass ciss_density_matrix(−χ) as the Hypothesis's recombination_state.
    """
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    triplet_zero = np.array([0, 1, 1, 0]) / np.sqrt(2)
    state = np.cos(ciss_angle_rad / 2) * singlet + np.sin(ciss_angle_rad / 2) * triplet_zero
    return np.outer(state, state).astype(complex)


# %% Operators and Hamiltonian (NumPy arrays; Hamiltonians in rad/µs)

def spin_matrices(spin):
    """[Sx, Sy, Sz] for a single spin of quantum number `spin` (dimensionless, ħ = 1), ordered m = +spin first."""
    return [qt.jmat(spin, axis).full() for axis in ("x", "y", "z")]


def spin_operators(hypothesis):
    """Spin operators on the full space, ordered electron 1 ⊗ electron 2 ⊗ nuclei (dimensionless, ħ = 1).

    Returns a list of [x, y, z] triples: entry 0 is electron 1, entry 1 is electron 2,
    entry 2 + n is hypothesis.nuclei[n]. Each operator is identity on every other spin.
    """
    spins = [0.5, 0.5] + [nucleus.spin for nucleus in hypothesis.nuclei]
    dimensions = [int(2 * spin + 1) for spin in spins]
    operators = []
    for position, spin in enumerate(spins):
        identity_before = np.eye(int(np.prod(dimensions[:position])))
        identity_after = np.eye(int(np.prod(dimensions[position + 1:])))
        operators.append([np.kron(np.kron(identity_before, matrix), identity_after)
                          for matrix in spin_matrices(spin)])
    return operators


def static_hamiltonian(hypothesis):
    """Static Hamiltonian in rad/µs: hyperfine + exchange J(S1·S2 − 1/4) + dipolar S1·D·S2 + Zeeman ω0(S1z + S2z),
    field along z.

    Hyperfine: for each nucleus, Σ_ab S_a A_ab I_b, with S the electron on that nucleus's radical.
    Dipolar: Σ_ab S1_a D_ab S2_b, D = hypothesis.dipolar_tensor_MHz (zero by default).
    MHz are converted to rad/µs here (angular_frequency), and nowhere else in the model.
    """
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

    spin_dot_product = sum(electron_1[a] @ electron_2[a] for a in range(3))
    exchange_term = angular_frequency(hypothesis.exchange_coupling_MHz) * (spin_dot_product - 0.25 * identity)
    dipolar = angular_frequency(hypothesis.dipolar_tensor_MHz)
    dipolar_term = sum(dipolar[a, b] * electron_1[a] @ electron_2[b] for a in range(3) for b in range(3))
    zeeman_term = angular_frequency(hypothesis.zeeman_frequency_MHz) * (electron_1[2] + electron_2[2])
    return hyperfine_term + exchange_term + dipolar_term + zeeman_term


def drive_operator(hypothesis):
    """Operator the RF drive couples to, S1x + S2x on the full space (dimensionless): the drive acts along x on both electrons.

    Eigenvalues 1, 0, 0, −1, each repeated over the nuclear states. It annihilates the singlet
    (total spin zero), so it never mixes singlet and triplet.
    """
    operators = spin_operators(hypothesis)
    return operators[0][0] + operators[1][0]


# %% Initial state and Haberkorn recombination

def nuclear_dimension(hypothesis):
    """Dimension of the nuclear part of the space, the product of (2I + 1) over the nuclei (1 if none)."""
    return int(np.prod([2 * nucleus.spin + 1 for nucleus in hypothesis.nuclei]))


def initial_density_matrix(hypothesis):
    """Full initial state ρ0 = ρ_electrons ⊗ (identity / nuclear dimension): electrons as given, nuclei fully mixed."""
    dimension = nuclear_dimension(hypothesis)
    return np.kron(hypothesis.electron_initial_state, np.eye(dimension) / dimension)


def singlet_projector(hypothesis):
    """Singlet projector P_S = |S⟩⟨S| ⊗ identity on the nuclei, on the full space."""
    return np.kron(singlet_density_matrix(), np.eye(nuclear_dimension(hypothesis)))


def recombination_projector(hypothesis):
    """P_R = |recombination state⟩⟨…| ⊗ identity on the nuclei, on the full space: the singlet projector P_S unless
    the hypothesis sets another recombination_state (e.g. the CISS one)."""
    return np.kron(hypothesis.recombination_state, np.eye(nuclear_dimension(hypothesis)))


def recombination_operator(hypothesis):
    """Haberkorn recombination operator K = k_S P_R + k_T (identity − P_R) in µs⁻¹ (P_R = P_S by default).

    Rates are probabilities per µs, so no 2π. When k_S = k_T = k, K = k × identity exactly.
    """
    projector = recombination_projector(hypothesis)
    rest = np.eye(projector.shape[0]) - projector
    return (hypothesis.singlet_recombination_rate_per_us * projector
            + hypothesis.triplet_recombination_rate_per_us * rest)


def effective_hamiltonian(hypothesis):
    """Non-Hermitian effective Hamiltonian H − iK/2 in rad/µs (rates in µs⁻¹ enter unchanged).

    Evolving ρ with it, dρ/dt = −i[H, ρ] − ½{K, ρ}, removes population in the recombination state (the singlet by
    default) at rate k_S and the rest at rate k_T (Haberkorn kinetics). The drive is added by the engine.
    """
    return static_hamiltonian(hypothesis) - 0.5j * recombination_operator(hypothesis)


# %% Orientation: turning the molecule relative to the field

def rotation_matrix(polar_angle_rad, azimuth_rad):
    """3×3 rotation that turns the z axis to the direction with polar angle θ and azimuth φ (radians).

    A tilt by θ about y, then a turn by φ about z: R = Rz(φ) Ry(θ), so R ẑ = (sin θ cos φ, sin θ sin φ, cos θ).
    """
    tilt = np.array([[np.cos(polar_angle_rad), 0, np.sin(polar_angle_rad)],
                     [0, 1, 0],
                     [-np.sin(polar_angle_rad), 0, np.cos(polar_angle_rad)]])
    turn = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                     [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
                     [0, 0, 1]])
    return turn @ tilt


def spin_half_rotation(polar_angle_rad, azimuth_rad):
    """2×2 matrix of rotation_matrix(θ, φ) acting on one spin-1/2 (basis ↑, ↓): exp(−iφ Sz) exp(−iθ Sy), closed form."""
    half_tilt, half_turn = polar_angle_rad / 2, azimuth_rad / 2
    spin_tilt = np.array([[np.cos(half_tilt), -np.sin(half_tilt)],
                          [np.sin(half_tilt), np.cos(half_tilt)]])
    spin_turn = np.diag([np.exp(-1j * half_turn), np.exp(1j * half_turn)])
    return spin_turn @ spin_tilt


def turned_hypothesis(hypothesis, rotation, spin_rotation, name):
    """Copy of the hypothesis with the molecule turned by the 3×3 rotation, spin_rotation being the same turn on
    one spin-1/2 (2×2). Turned: every hyperfine tensor and the dipolar tensor (T → R T Rᵀ), the electron start state
    and the recombination state (ρ → U ρ U†, U the turn on both electrons; the singlet has no direction, so it is
    unchanged). Not turned: the field (lab z), the drive (lab x), and the exchange coupling (no direction).
    """
    nuclei = tuple(dataclasses.replace(nucleus,
                                       hyperfine_tensor_MHz=rotation @ nucleus.hyperfine_tensor_MHz @ rotation.T)
                   for nucleus in hypothesis.nuclei)
    both_electrons = np.kron(spin_rotation, spin_rotation)
    return hypothesis.variant(
        name=name, nuclei=nuclei, dipolar_tensor_MHz=rotation @ hypothesis.dipolar_tensor_MHz @ rotation.T,
        electron_initial_state=both_electrons @ hypothesis.electron_initial_state @ both_electrons.conj().T,
        recombination_state=both_electrons @ hypothesis.recombination_state @ both_electrons.conj().T)


def oriented_hypothesis(hypothesis, polar_angle_rad, azimuth_rad):
    """Copy of the hypothesis with the molecule turned so that its z axis points along (θ, φ) in the lab (radians).

    The hypothesis is read as given with the molecule's axis along the field (lab z); see turned_hypothesis for
    what turns. For a molecule symmetric about its z axis this covers every field direction. For one that is not
    (e.g. flavin_tryptophan_hypothesis), the field seen by the molecule always lies in its xz plane, whatever φ:
    use field_direction_hypothesis to scan field directions.
    """
    return turned_hypothesis(hypothesis, rotation_matrix(polar_angle_rad, azimuth_rad),
                             spin_half_rotation(polar_angle_rad, azimuth_rad),
                             f"{hypothesis.name}, axis at theta={polar_angle_rad:.4g}, phi={azimuth_rad:.4g} rad")


def field_direction_hypothesis(hypothesis, polar_angle_rad, azimuth_rad):
    """Copy of the hypothesis with the field along the direction (θ, φ) of the molecule's own frame (radians).

    n = (sin θ cos φ, sin θ sin φ, cos θ) in the frame the tensors are given in. The molecule is turned by the
    reverse of rotation_matrix(θ, φ), which brings n onto the field (lab z). The drive (lab x) then points along
    (cos θ cos φ, cos θ sin φ, −sin θ) in the molecule's frame: perpendicular to the field, towards increasing θ
    ("south" on a globe whose north pole is the molecule's z axis). This choice of drive direction is a convention.
    """
    return turned_hypothesis(hypothesis, rotation_matrix(polar_angle_rad, azimuth_rad).T,
                             spin_half_rotation(polar_angle_rad, azimuth_rad).conj().T,
                             f"{hypothesis.name}, field at theta={polar_angle_rad:.4g}, phi={azimuth_rad:.4g} rad")


# %% Presets: ready-made hypotheses, so a script does not fill in every field by hand

def baseline_hypothesis(delta):
    """Baseline pair: one spin-1 nucleus on radical 1 with axial hyperfine anisotropy delta, singlet-born.

    Parameter values come from DISCO (old repo) and may rely on stale placeholder parameters.
    Use for sanity checks against the old results, not as physically motivated values.
    """
    # DISCO values (old repo, All DISCO/DISCO.py lines 149-152); may be stale placeholders.
    hyperfine_coupling_MHz = 28.0
    exchange_coupling_MHz = 1.0
    zeeman_frequency_MHz = 28.0
    recombination_rate_per_us = 1.0

    nucleus = Nucleus(radical=1, spin=1,
                      hyperfine_tensor_MHz=axial_hyperfine_tensor_MHz(hyperfine_coupling_MHz, delta))
    return Hypothesis(
        name=f"baseline, delta={delta}",
        nuclei=(nucleus,),
        exchange_coupling_MHz=exchange_coupling_MHz,
        zeeman_frequency_MHz=zeeman_frequency_MHz,
        electron_initial_state=singlet_density_matrix(),
        singlet_recombination_rate_per_us=recombination_rate_per_us,
        triplet_recombination_rate_per_us=recombination_rate_per_us,
    )


def flavin_tryptophan_hypothesis(ciss_angle_rad, recombination_rate_per_us, forward_rate_per_us, field_strength_uT):
    """Daniel Kattnig's flavin–tryptophan pair: one nitrogen-14 on each radical (FN5 on electron 1, WN1 on
    electron 2), electron-electron dipolar coupling EED, no exchange. Dimension 2 × 2 × 3 × 3 = 36.

    Tensors are in the molecule's frame (the field is turned through it with field_direction_hypothesis).
    CISS angle χ (rad): start cos(χ/2)|S⟩ + sin(χ/2)|T0⟩ along the molecule's z axis, recombination through
    cos(χ/2)|S⟩ − sin(χ/2)|T0⟩ (Daniel's recombination_projector_ciss); χ = 0 is the ordinary singlet pair.
    Rates (µs⁻¹), read as K = k_r P_R + k_f × identity: recombination at k_r through P_R, and a spin-independent
    forward reaction (separation) at k_f, i.e. k_S = k_r + k_f and k_T = k_f here.
    """
    # Tensors from Daniel Kattnig (message received by 2026-10-15), given in rad/µs; /2π once here gives MHz.
    # FN5: flavin N5, WN1: tryptophan N1, EED: electron-electron dipolar coupling (axial about the molecule's z).
    FN5_rad_per_us = np.array([[79.48092385, 137.59553919, 56.68450732],
                               [137.59553919, 181.20322632, 79.23527745],
                               [56.68450732, 79.23527745, 15.748773]])
    WN1_rad_per_us = np.array([[51.01430479, 16.46786544, 90.57082545],
                               [16.46786544, -6.75181867, 23.32643811],
                               [90.57082545, 23.32643811, 125.59269667]])
    EED_rad_per_us = np.diag([6.00327982e+01, 6.00327982e+01, -1.20065596e+02])
    # Assumptions to confirm with Daniel: exchange J = 0 (he gave only EED; on 9 Oct: neglect exchange for separated
    # radicals); the rates reading above; the CISS axis along the molecule's z (the EED axis); electron 1 the donor.
    exchange_coupling_MHz = 0.0

    nuclei = (Nucleus(radical=1, spin=1, hyperfine_tensor_MHz=FN5_rad_per_us / (2 * np.pi)),
              Nucleus(radical=2, spin=1, hyperfine_tensor_MHz=WN1_rad_per_us / (2 * np.pi)))
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
