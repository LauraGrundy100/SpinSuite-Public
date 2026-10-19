"""Regression tests: the tidied reference engine must reproduce the old repo's recorded numbers.

Sources (old repo):
    All DISCO/DISCO_run_1/summary/summary_report.txt          static yields, printed to 6 decimal places
    All DISCO/DISCO_run_1/run3_discriminate/run3_results.npz  optimised drive and its yields, full precision
    Main Experimental scripts/PolyDiscrim_results/PolyDiscrim_results.npz
        monochromatic scan, linspace(0.5, 60, 1200) MHz, full precision

Every expected number here comes from DISCO (old repo) and may rely on stale placeholder
parameters. Use them only as sanity checks for the port, never as values for new work.

The reference engine now solves at tight tolerances, while the old code used QuTiP's defaults.
So these numbers are matched within the old engine's own error (about 3e-5 or less), not exactly.

TEMPORARY: these tests only check that the move from the old repo was correct.
They are archived once the new repo is complete (see CLAUDE.md).
"""

import numpy as np
import pytest

from reference_engine import (drive_coefficient, linf_scale, singlet_population,
                              singlet_yield, spin_operators, static_hamiltonian)

# Old DISCO setup (old repo, All DISCO/DISCO.py lines 149-163); may rely on stale placeholder parameters.
HYPERFINE_COUPLING_MHz = 28.0
EXCHANGE_COUPLING_MHz = 1.0
ZEEMAN_FREQUENCY_MHz = 28.0
RECOMBINATION_RATE_PER_US = 1.0
AMPLITUDE_CAP_MHz = 0.25 * HYPERFINE_COUPLING_MHz
DELTA_A = 0.0
DELTA_B = 0.3
TIMES_US = np.linspace(0, 6, 2000)

# The old engine's own numerical error in D is about 1e-5 (CLAUDE.md)
TOLERANCE = 1e-4

# Optimised discrimination drive from DISCO's old run3_results.npz; may rely on stale placeholder parameters.
# Frequencies were saved in rad/µs and are converted here to MHz (divided by 2π).
RUN3_FREQUENCIES_MHz = [31.278903364378777, 28.331912643861415, 0.29370083696399624, 10.865639062914235]
RUN3_RAW_AMPLITUDES = [-0.360134557742187, 142.36189395265416, -0.34001651040795267, 2476.2233989893016]
RUN3_PHASES_RAD = [9.91228216421405, 0.8418697480570057, 0.012293007456405332, -0.1589099776003636]
RUN3_SCALE = 0.01679517530753102

# Monochromatic scan points from the old PolyDiscrim_results.npz (DISCO setup): (frequency MHz, Y_A, Y_B, D).
# May rely on stale placeholder parameters.
# The scan did not save its phase. Phase 0 with raw amplitude 1 reproduces the saved yields
# to about 1e-9, and phase π/2 does not, so the scan used phase 0.
# 49.98 MHz is the best point of that scan (which stopped at 60 MHz); 11.02 MHz gives D ≈ 0.049.
MONO_SCAN_POINTS = [
    (11.020433694745622, 0.33586049336315105, 0.38505175783272383, 0.04919126446957278),
    (49.97581317764804, 0.3307358190377631, 0.39142785974807115, 0.06069204071030804),
]


def yield_for(delta, coefficient=None):
    """Singlet yield of the old DISCO model at anisotropy delta, with an optional drive along x."""
    operators = spin_operators()
    hamiltonian = static_hamiltonian(operators, HYPERFINE_COUPLING_MHz, delta,
                                     EXCHANGE_COUPLING_MHz, ZEEMAN_FREQUENCY_MHz)
    if coefficient is not None:
        hamiltonian = [hamiltonian, [operators["S1x"] + operators["S2x"], coefficient]]
    population = singlet_population(hamiltonian, TIMES_US)
    return singlet_yield(population, TIMES_US, RECOMBINATION_RATE_PER_US)


def test_static_yields_match_old_repo():
    yield_A = yield_for(DELTA_A)
    yield_B = yield_for(DELTA_B)
    # From DISCO (DISCO_run_1 summary_report.txt); may rely on stale placeholder parameters.
    assert abs(yield_A - 0.383468) < TOLERANCE
    assert abs(yield_B - 0.399011) < TOLERANCE
    assert abs(abs(yield_A - yield_B) - 0.015544) < TOLERANCE


def test_linf_scale_matches_old_run3():
    scale = linf_scale(RUN3_FREQUENCIES_MHz, RUN3_RAW_AMPLITUDES, RUN3_PHASES_RAD, AMPLITUDE_CAP_MHz)
    assert np.isclose(scale, RUN3_SCALE, rtol=1e-10, atol=0)


def test_optimised_drive_matches_old_run3():
    coefficient = drive_coefficient(RUN3_FREQUENCIES_MHz, RUN3_RAW_AMPLITUDES, RUN3_PHASES_RAD, AMPLITUDE_CAP_MHz)
    yield_A = yield_for(DELTA_A, coefficient)
    yield_B = yield_for(DELTA_B, coefficient)
    # From DISCO (run3_results.npz); may rely on stale placeholder parameters.
    assert abs(yield_A - 0.3135582140564846) < TOLERANCE
    assert abs(yield_B - 0.37390918357426545) < TOLERANCE
    assert abs(abs(yield_A - yield_B) - 0.06035096951778085) < TOLERANCE


@pytest.mark.parametrize("frequency_MHz, old_yield_A, old_yield_B, old_gap", MONO_SCAN_POINTS)
def test_monochromatic_drive_matches_old_scan(frequency_MHz, old_yield_A, old_yield_B, old_gap):
    coefficient = drive_coefficient([frequency_MHz], [1.0], [0.0], AMPLITUDE_CAP_MHz)
    yield_A = yield_for(DELTA_A, coefficient)
    yield_B = yield_for(DELTA_B, coefficient)
    assert abs(yield_A - old_yield_A) < TOLERANCE
    assert abs(yield_B - old_yield_B) < TOLERANCE
    assert abs(abs(yield_A - yield_B) - old_gap) < TOLERANCE
