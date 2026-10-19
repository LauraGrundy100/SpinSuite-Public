"""Tests for control.py.

The last section compares against numbers from the old repo. It is TEMPORARY: it only checks that the
move from the old repo was correct, and is archived once the new repo is complete (see CLAUDE.md).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import reference_engine
from control import (Drive, action_spectrum, action_spectrum_seeds, bhattacharyya_cost, contrast_cost,
                     discrimination_cost, disruption_cost, drive_rad_per_us, l2_drive, linf_drive, load_search_state,
                     optimise_drive, random_drive_parameters, random_starting_frequencies, waveform)
from engine import (averaged_yields_together, bhattacharyya_distance, field_direction_yields, orientation_grid,
                    outcome_probabilities, yields)
from model import baseline_hypothesis, ciss_density_matrix
from results import read_npz
from test_engine import RUN3_FREQUENCIES_MHz, RUN3_PHASES_RAD, RUN3_RAW_AMPLITUDES
from test_model import flavin_tryptophan_test_pairs

# The old repo's peak check for L-infinity normalisation: 0–12 µs, 20000 points (All DISCO/DISCO.py line 254).
OLD_PEAK_CHECK_TIMES_US = np.linspace(0, 12, 20000)
AMPLITUDE_CAP_MHz = 7.0  # 0.25 × A_iso, DISCO value; may rely on stale placeholder parameters


# %% Drives and normalisation

def test_waveform_matches_hand_calculation():
    # At t = 0.0125 µs a 20 MHz sine has done a quarter cycle (2π × 20 × 0.0125 = π/2), so sin = 1.
    # A 40 MHz sine has done half a cycle, so with phase π/2 it gives sin(π + π/2) = −1.
    value = waveform(0.0125, [20.0, 40.0], [3.0, 2.0], [0.0, np.pi / 2])
    assert np.isclose(value, 3.0 - 2.0, rtol=0, atol=1e-12)


def test_linf_drive_peak_equals_cap():
    drive = linf_drive([28.0, 11.0, 5.3], [1.0, -0.4, 2.5], [0.1, 0.7, -1.2], AMPLITUDE_CAP_MHz,
                       OLD_PEAK_CHECK_TIMES_US)
    peak = np.max(np.abs(waveform(OLD_PEAK_CHECK_TIMES_US, drive.frequencies_MHz, drive.amplitudes_MHz,
                                  drive.phases_rad)))
    assert np.isclose(peak, AMPLITUDE_CAP_MHz, rtol=1e-12, atol=0)


def test_linf_drive_ignores_overall_size_of_raw_amplitudes():
    small = linf_drive([28.0, 11.0], [1.0, -0.4], [0.1, 0.7], AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US)
    large = linf_drive([28.0, 11.0], [3.0, -1.2], [0.1, 0.7], AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US)
    assert np.allclose(small.amplitudes_MHz, large.amplitudes_MHz, rtol=1e-12, atol=0)


def test_l2_drive_fixes_root_sum_of_squares():
    # Four equal components share the cap: each gets 7 / √4 = 3.5 MHz.
    equal = l2_drive([28.0, 27.0, 14.0, 13.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], AMPLITUDE_CAP_MHz)
    assert np.allclose(equal.amplitudes_MHz, 3.5, rtol=1e-12, atol=0)
    unequal = l2_drive([28.0, 11.0, 5.3], [1.0, -0.4, 2.5], [0.1, 0.7, -1.2], AMPLITUDE_CAP_MHz)
    assert np.isclose(np.sqrt(np.sum(unequal.amplitudes_MHz ** 2)), AMPLITUDE_CAP_MHz, rtol=1e-12, atol=0)


def test_single_sine_gets_the_cap_under_both_normalisations():
    # One sine has peak = amplitude, so both caps agree. L-infinity samples the peak, so it matches only closely
    # (measured: about 1e-10 relative at 11.3 MHz on the old 0–12 µs grid).
    assert np.isclose(l2_drive([11.3], [0.2], [0.0], AMPLITUDE_CAP_MHz).amplitudes_MHz[0], AMPLITUDE_CAP_MHz,
                      rtol=1e-12, atol=0)
    linf_amplitude = linf_drive([11.3], [0.2], [0.0], AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US).amplitudes_MHz[0]
    assert np.isclose(linf_amplitude, AMPLITUDE_CAP_MHz, rtol=1e-4, atol=0)


def test_zero_drive_gives_zero_amplitudes():
    for drive in (linf_drive([28.0], [0.0], [0.0], AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US),
                  l2_drive([28.0], [0.0], [0.0], AMPLITUDE_CAP_MHz)):
        assert np.all(drive.amplitudes_MHz == 0.0)


def test_drive_function_matches_reference_engine_drive_code():
    # The reference engine holds the old drive code (L-infinity, 0–12 µs check). The run-3 drive and the
    # 11.02 MHz point come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 6, 2000)
    for frequencies_MHz, raw_amplitudes, phases_rad in [
            (RUN3_FREQUENCIES_MHz, RUN3_RAW_AMPLITUDES, RUN3_PHASES_RAD),
            ([11.020433694745622], [1.0], [0.0])]:
        drive = linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US)
        reference = reference_engine.drive_coefficient(frequencies_MHz, raw_amplitudes, phases_rad,
                                                       AMPLITUDE_CAP_MHz)
        assert np.allclose(drive_rad_per_us(drive)(times_us), reference(times_us), rtol=1e-10, atol=1e-10)


def test_drive_function_accepts_a_single_time():
    # The density-matrix reference calls the drive one time at a time; the engine passes arrays.
    drive = Drive(np.array([20.0]), np.array([1.0]), np.array([0.0]))
    assert np.isclose(drive_rad_per_us(drive)(0.0125), 2 * np.pi, rtol=1e-12, atol=0)


# %% Starting frequencies: action-spectrum seeding

def test_seeds_follow_score_order():
    # Worked example: scores 0.1, 1.0, 0.3, 0.7 at 0.5, 1.5, 2.5, 3.5 MHz give seeds 1.5, 3.5, 2.5 MHz.
    seeds = action_spectrum_seeds(np.array([0.5, 1.5, 2.5, 3.5]), np.array([0.1, 1.0, 0.3, 0.7]), 3, 0.5)
    assert seeds.tolist() == [1.5, 3.5, 2.5]


def test_seeds_skip_bins_too_close_to_a_chosen_one():
    # 0.25 MHz bins with a 0.5 MHz minimum separation: 1.125 is next in score but only 0.25 MHz from 1.375.
    centres = np.array([0.875, 1.125, 1.375, 1.625])
    seeds = action_spectrum_seeds(centres, np.array([0.2, 0.9, 1.0, 0.1]), 2, 0.5)
    assert seeds.tolist() == [1.375, 0.875]


def test_random_starting_frequencies_are_in_range_reproducible_and_unlike_the_nudges():
    frequencies_MHz = random_starting_frequencies(1, 4, 40.0)
    assert len(frequencies_MHz) == 4 and np.all((frequencies_MHz >= 0) & (frequencies_MHz <= 40.0))
    assert np.array_equal(frequencies_MHz, random_starting_frequencies(1, 4, 40.0))  # same seed, same start
    assert not np.allclose(frequencies_MHz, random_starting_frequencies(2, 4, 40.0))  # another seed, another start
    # The optimiser's nudges for seed 1 come from the older generator; mapped onto 0–40 MHz they differ.
    assert not np.allclose(frequencies_MHz, np.random.RandomState(1).uniform(0.0, 40.0, size=4))


def test_random_drives_are_in_range_and_reproducible_and_capped_like_any_drive():
    first = random_drive_parameters(np.random.default_rng(5), 4, 40.0)
    frequencies_MHz, raw_amplitudes, phases_rad = first
    assert np.all((frequencies_MHz >= 0) & (frequencies_MHz <= 40.0))
    assert np.all((raw_amplitudes >= 0) & (raw_amplitudes <= 1.0))
    assert np.all((phases_rad >= 0) & (phases_rad <= 2 * np.pi))
    again = random_drive_parameters(np.random.default_rng(5), 4, 40.0)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))  # same seed, same drive
    drive = linf_drive(*first, AMPLITUDE_CAP_MHz, OLD_PEAK_CHECK_TIMES_US)
    peak = np.max(np.abs(waveform(OLD_PEAK_CHECK_TIMES_US, drive.frequencies_MHz, drive.amplitudes_MHz, drive.phases_rad)))
    assert abs(peak - AMPLITUDE_CAP_MHz) < 1e-12  # after the cap, as strong as an optimised drive


def test_too_few_separated_bins_is_an_error():
    with pytest.raises(ValueError):
        action_spectrum_seeds(np.array([0.5, 1.5]), np.array([1.0, 0.5]), 3, 0.5)


def test_action_spectrum_is_scaled_to_one_and_ignores_recombination():
    # The spectrum uses the Hermitian H only (as the old code did), so changing k_S and k_T changes nothing.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    hypothesis = baseline_hypothesis(0.3)
    centres_MHz, scores = action_spectrum(hypothesis, 1.0, 40.0)
    assert np.isclose(scores.max(), 1.0) and np.all(scores >= 0)
    assert np.allclose(centres_MHz, np.arange(40) + 0.5)
    _, scores_unequal_rates = action_spectrum(
        hypothesis.variant(singlet_recombination_rate_per_us=2.0, triplet_recombination_rate_per_us=0.5), 1.0, 40.0)
    assert np.array_equal(scores, scores_unequal_rates)


def test_action_spectrum_over_the_aligned_direction_alone_is_unchanged():
    # "Averaging" over one direction, along the field, with weight 1 is the molecule as given: identical scores.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    hypothesis = baseline_hypothesis(0.3)
    _, aligned = action_spectrum(hypothesis, 1.0, 40.0)
    _, one_direction = action_spectrum(hypothesis, 1.0, 40.0, (np.array([0.0]), np.array([0.0]), np.array([1.0])))
    assert np.array_equal(aligned, one_direction)


def test_averaged_action_spectrum_differs_from_aligned():
    # Tilted molecules have other transition frequencies, so the sample's spectrum must differ from the aligned one
    # (a check that the averaging is not silently skipped). Still scaled so the largest score is 1.
    hypothesis = baseline_hypothesis(0.3)
    _, aligned = action_spectrum(hypothesis, 1.0, 40.0)
    _, averaged = action_spectrum(hypothesis, 1.0, 40.0, orientation_grid(11))
    assert np.isclose(averaged.max(), 1.0)
    assert np.max(np.abs(averaged - aligned)) > 0.1


# %% Scoring a drive

def test_costs_agree_with_engine_yields():
    # A coarse step (one substep per output interval) is enough here: both sides use the same engine call.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 6, 2000)
    max_step_us = 0.003
    drive = Drive(np.array([28.0, 11.0]), np.array([5.0, 3.0]), np.array([0.0, 0.4]))
    hypothesis_A, hypothesis_B = baseline_hypothesis(0.0), baseline_hypothesis(0.3)
    yield_A, _ = yields(hypothesis_A, drive_rad_per_us(drive), times_us, max_step_us)
    yield_B, _ = yields(hypothesis_B, drive_rad_per_us(drive), times_us, max_step_us)
    assert disruption_cost(hypothesis_A, drive, times_us, max_step_us) == yield_A
    # Discrimination simulates both pairs in one pass, which agrees with separate passes to rounding (about 2e-15).
    gap = abs(yield_A - yield_B)
    assert abs(discrimination_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us) + gap) < 1e-12
    assert abs(discrimination_cost(hypothesis_B, hypothesis_A, drive, times_us, max_step_us) + gap) < 1e-12
    # The Bhattacharyya cost is −√B built from the same yields, all three outcomes, either order.
    yields_A, yields_B = (yields(hypothesis, drive_rad_per_us(drive), times_us, max_step_us)
                          for hypothesis in (hypothesis_A, hypothesis_B))
    distance = bhattacharyya_distance(outcome_probabilities(*yields_A), outcome_probabilities(*yields_B))
    assert abs(bhattacharyya_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us) + np.sqrt(distance)) < 1e-12
    assert abs(bhattacharyya_cost(hypothesis_B, hypothesis_A, drive, times_us, max_step_us) + np.sqrt(distance)) < 1e-12
    # The same hypothesis twice cannot be told apart, so the cost is zero.
    assert bhattacharyya_cost(hypothesis_A, hypothesis_A, drive, times_us, max_step_us) == pytest.approx(0.0, abs=1e-7)


def test_costs_with_orientations_use_the_averaged_yields():
    # With an orientation grid, every cost is built from averaged_yields_together. Short window, 6 directions.
    # Baseline parameters come from DISCO (old repo) and may rely on stale placeholder parameters.
    times_us = np.linspace(0, 1, 201)
    max_step_us = 0.0015
    orientations = orientation_grid(3)
    drive = Drive(np.array([28.0, 11.0]), np.array([5.0, 3.0]), np.array([0.0, 0.4]))
    hypothesis_A = baseline_hypothesis(0.3)
    hypothesis_B = hypothesis_A.variant(electron_initial_state=ciss_density_matrix(0.25 * np.pi))
    yields_A, yields_B = averaged_yields_together([hypothesis_A, hypothesis_B], drive_rad_per_us(drive), times_us,
                                                  max_step_us, orientations)
    distance = bhattacharyya_distance(outcome_probabilities(*yields_A), outcome_probabilities(*yields_B))
    assert disruption_cost(hypothesis_A, drive, times_us, max_step_us, orientations) == pytest.approx(
        yields_A[0], abs=1e-14)
    gap_cost = discrimination_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations)
    assert gap_cost == pytest.approx(-abs(yields_A[0] - yields_B[0]), abs=1e-14)
    noise_cost = bhattacharyya_cost(hypothesis_A, hypothesis_B, drive, times_us, max_step_us, orientations)
    assert noise_cost == pytest.approx(-np.sqrt(distance), abs=1e-12)
    # And the average really differs from the aligned molecule (it would be exactly 0 if the averaging were skipped).
    # Measured difference in this short window: 2.8e-4.
    assert abs(disruption_cost(hypothesis_A, drive, times_us, max_step_us) - yields_A[0]) > 1e-5


def test_contrast_cost_is_minus_the_difference_in_contrast():
    # Built by hand from the per-direction yields: contrast = highest − lowest of k_r/k_S × Y_S over the directions.
    times_us = np.linspace(0, 1, 201)
    max_step_us = 0.0015
    field_directions = (np.array([0.3, 1.5]), np.array([0.0, 2.0]))
    drive = Drive(np.array([12.0]), np.array([2.0]), np.array([0.0]))
    classical, ciss = flavin_tryptophan_test_pairs()
    contrasts = []
    for hypothesis, rate_share in ((classical, 1 / 2), (ciss, 10 / 11)):
        singlet_yields, _ = field_direction_yields(hypothesis, drive_rad_per_us(drive), times_us, max_step_us,
                                                   field_directions)
        contrasts.append(rate_share * abs(singlet_yields[0] - singlet_yields[1]))
    cost = contrast_cost(classical, ciss, drive, times_us, max_step_us, field_directions)
    assert cost == pytest.approx(-abs(contrasts[0] - contrasts[1]), abs=1e-14)
    assert contrast_cost(ciss, classical, drive, times_us, max_step_us, field_directions) == pytest.approx(cost, abs=1e-15)


# %% Optimiser

def toy_cost(frequencies_MHz, raw_amplitudes, phases_rad):
    """Made-up cost with no physics, lowest (zero) at frequencies 10 and 20 MHz, amplitudes 2 and phases 0."""
    return (np.sum((frequencies_MHz - [10.0, 20.0]) ** 2) + np.sum((raw_amplitudes - 2.0) ** 2)
            + np.sum(phases_rad ** 2))


def test_optimiser_only_keeps_improvements():
    result = optimise_drive(toy_cost, [12.0, 17.0], 10, 5.0, 200, 123, 10)
    history = result.cost_history
    assert len(history) == 11 and len(result.accepted) == 10 and len(result.evaluations) == 10
    assert np.all(np.diff(history) <= 0)  # the best cost never gets worse
    assert np.array_equal(result.accepted, np.diff(history) < 0)  # kept exactly when it improved
    assert history[-1] == toy_cost(result.frequencies_MHz, result.raw_amplitudes, result.phases_rad)
    assert history[-1] < history[0]


def test_optimiser_records_every_nelder_mead_cost():
    # One recorded cost per call; each round's lowest recorded cost is that round's candidate: equal to the new best
    # when kept, and no better than the old best when not kept. The record never changes the search itself.
    result = optimise_drive(toy_cost, [12.0, 17.0], 10, 5.0, 200, 123, 10)
    assert len(result.nelder_mead_costs) == np.sum(result.evaluations)
    round_starts = np.concatenate([[0], np.cumsum(result.evaluations)])
    for round_number in range(1, len(result.accepted) + 1):
        lowest = result.nelder_mead_costs[round_starts[round_number - 1]:round_starts[round_number]].min()
        if result.accepted[round_number - 1]:
            assert lowest == result.cost_history[round_number]
        else:
            assert lowest >= result.cost_history[round_number - 1]


def test_old_checkpoint_without_recorded_costs_loads_with_nan(tmp_path):
    # Checkpoints from before 2026-10-14 (version 1) have no nelder_mead_costs: they load as NaN, never invented.
    checkpoint_path = tmp_path / "seed_123.npz"
    optimise_drive(toy_cost, [12.0, 17.0], 3, 5.0, 50, 123, 3, checkpoint_path)
    saved = read_npz(checkpoint_path)
    del saved["nelder_mead_costs"]
    np.savez(checkpoint_path, **{**saved, "checkpoint_format_version": 1})
    state = load_search_state(checkpoint_path)
    assert len(state.nelder_mead_costs) == np.sum(state.evaluations)
    assert np.all(np.isnan(state.nelder_mead_costs))


def test_optimiser_is_reproducible_from_its_seed():
    first = optimise_drive(toy_cost, [12.0, 17.0], 5, 5.0, 50, 7, 5)
    second = optimise_drive(toy_cost, [12.0, 17.0], 5, 5.0, 50, 7, 5)
    assert np.array_equal(first.frequencies_MHz, second.frequencies_MHz)
    assert np.array_equal(first.cost_history, second.cost_history)


def test_optimiser_stops_after_patience_runs_out():
    # A cost that never changes can never improve, so with patience 3 the search stops after 3 of 10 rounds.
    result = optimise_drive(lambda frequencies, amplitudes, phases: 1.0, [12.0, 17.0], 10, 5.0, 5, 123, 3)
    assert len(result.cost_history) == 4 and not np.any(result.accepted)


class SimulatedCrash(Exception):
    """Raised by crashing_cost to stand in for a power cut or Ctrl+C part-way through a search."""


def crashing_cost(crash_at_call):
    """toy_cost, except that call number crash_at_call raises SimulatedCrash."""
    calls = {"count": 0}

    def cost(frequencies_MHz, raw_amplitudes, phases_rad):
        calls["count"] += 1
        if calls["count"] == crash_at_call:
            raise SimulatedCrash
        return toy_cost(frequencies_MHz, raw_amplitudes, phases_rad)

    return cost


def test_interrupted_search_resumes_to_exactly_the_uninterrupted_result(tmp_path):
    uninterrupted = optimise_drive(toy_cost, [12.0, 17.0], 10, 5.0, 200, 123, 10)
    total_calls = 1 + int(np.sum(uninterrupted.evaluations))  # the start cost, then every round's calls
    checkpoint_path = tmp_path / "seed_123.npz"
    with pytest.raises(SimulatedCrash):  # crash half-way through the calls, so part-way through some round
        optimise_drive(crashing_cost(total_calls // 2), [12.0, 17.0], 10, 5.0, 200, 123, 10, checkpoint_path)
    rounds_saved = len(load_search_state(checkpoint_path).accepted)
    assert 0 < rounds_saved < 10  # the crash really did happen part-way through the search

    resumed = optimise_drive(toy_cost, [12.0, 17.0], 10, 5.0, 200, 123, 10, checkpoint_path)
    for name in ("frequencies_MHz", "raw_amplitudes", "phases_rad", "cost_history", "accepted", "evaluations",
                 "nelder_mead_costs"):
        assert np.array_equal(getattr(resumed, name), getattr(uninterrupted, name)), name  # exactly equal


def test_finished_search_returns_its_result_without_new_cost_calls(tmp_path):
    checkpoint_path = tmp_path / "seed_123.npz"
    finished = optimise_drive(toy_cost, [12.0, 17.0], 4, 5.0, 50, 123, 4, checkpoint_path)
    again = optimise_drive(crashing_cost(1), [12.0, 17.0], 4, 5.0, 50, 123, 4, checkpoint_path)  # any call would crash
    assert np.array_equal(again.cost_history, finished.cost_history)
    assert np.array_equal(again.frequencies_MHz, finished.frequencies_MHz)


def toy_search(random_seed):
    """A short optimiser run on the toy cost; module-level so a separate process can run it."""
    return optimise_drive(toy_cost, [12.0, 17.0], 3, 5.0, 30, random_seed, 3).frequencies_MHz


def test_searches_in_separate_processes_match_one_after_another():
    # experiments/disruption_sweep.py runs each anisotropy in its own process; results must not depend on that.
    random_seeds = [1, 2, 3]
    with ProcessPoolExecutor(max_workers=3) as pool:
        side_by_side = list(pool.map(toy_search, random_seeds))
    for random_seed, frequencies_MHz in zip(random_seeds, side_by_side):
        assert np.array_equal(frequencies_MHz, toy_search(random_seed))


def test_frequency_nudges_match_old_random_numbers():
    # The old code seeded NumPy's global generator and drew nudges in rad/µs (All DISCO/DISCO.py lines 355, 368).
    # The optimiser draws in MHz from a private generator of the same kind, so the nudges agree up to rounding.
    np.random.seed(123)
    old_nudges_MHz = np.random.uniform(-5 * 2 * np.pi, 5 * 2 * np.pi, size=4) / (2 * np.pi)
    new_nudges_MHz = np.random.RandomState(123).uniform(-5.0, 5.0, size=4)
    assert np.allclose(new_nudges_MHz, old_nudges_MHz, rtol=0, atol=1e-12)


def test_short_real_search_improves_disruption():
    # Smoke test: two super-iterations, at most 10 Nelder-Mead iterations each, on a short coarse window.
    # Checks that the pieces fit together, not the physics. Baseline parameters come from DISCO (old repo)
    # and may rely on stale placeholder parameters.
    hypothesis = baseline_hypothesis(0.0)
    times_us = np.linspace(0, 3, 1000)
    centres_MHz, scores = action_spectrum(hypothesis, 1.0, 40.0)

    def cost(frequencies_MHz, raw_amplitudes, phases_rad):
        drive = linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, AMPLITUDE_CAP_MHz, times_us)
        return disruption_cost(hypothesis, drive, times_us, 0.003)

    result = optimise_drive(cost, action_spectrum_seeds(centres_MHz, scores, 4, 0.5), 2, 5.0, 10, 123, 2)
    assert np.all(np.diff(result.cost_history) <= 0)
    static_cost = disruption_cost(hypothesis, Drive(np.array([]), np.array([]), np.array([])), times_us, 0.003)
    assert result.cost_history[-1] < static_cost  # driving lowers the singlet yield


# %% TEMPORARY: comparisons with old-repo numbers (archive once the new repo is complete)

# Step for these comparisons: Laura's experiment default (1.5 ns). Its error, about 2e-7 for drives up to
# about 50 MHz, is far below the 1e-4 tolerance used against the old engine.
OLD_COMPARISON_MAX_STEP_US = 0.0015
OLD_ENGINE_TOLERANCE = 1e-4  # the old engine's own error in D is about 1e-5 (CLAUDE.md)

# APEX δ = 0.0 run, from All APEX/APEX_Run_1/delta_0.0/deltadelta_00_results.npz (frequencies converted to MHz).
# APEX used DISCO's parameters, which may be stale placeholders, and a 0–10 µs window with 3000 points.
APEX_TIMES_US = np.linspace(0, 10, 3000)
APEX_FREQUENCIES_MHz = [28.30931901484361, 26.421175181941507, 12.931780161508694, 15.790497073840415]
APEX_RAW_AMPLITUDES = [9.51651961532369, 9.87517238804268, -1.4112197563380529, -0.009655828567722268]
APEX_PHASES_RAD = [0.07490924160904572, 0.0019325387251093212, -0.11003753639195554, 0.018163264221694054]
APEX_SCALE = 2.1159833414528917  # rad/µs per unit raw amplitude, as the old code stored it


def test_old_run3_drive_gives_old_gap():
    # Run-3 drive and yields from DISCO (run3_results.npz); may rely on stale placeholder parameters.
    # Measured difference: about 8e-6 in Y_B and D, 9e-8 in Y_A.
    drive = linf_drive(RUN3_FREQUENCIES_MHz, RUN3_RAW_AMPLITUDES, RUN3_PHASES_RAD, AMPLITUDE_CAP_MHz,
                       OLD_PEAK_CHECK_TIMES_US)
    times_us = np.linspace(0, 6, 2000)
    yield_A, _ = yields(baseline_hypothesis(0.0), drive_rad_per_us(drive), times_us, OLD_COMPARISON_MAX_STEP_US)
    yield_B, _ = yields(baseline_hypothesis(0.3), drive_rad_per_us(drive), times_us, OLD_COMPARISON_MAX_STEP_US)
    assert abs(yield_A - 0.3135582140564846) < OLD_ENGINE_TOLERANCE
    assert abs(yield_B - 0.37390918357426545) < OLD_ENGINE_TOLERANCE
    assert abs(abs(yield_A - yield_B) - 0.06035096951778085) < OLD_ENGINE_TOLERANCE


def test_old_apex_drive_gives_old_scale_and_yield():
    # APEX δ = 0.0 optimised drive and yield (deltadelta_00_results.npz); may rely on stale placeholder parameters.
    # Measured difference in the yield: about 3e-6.
    drive = linf_drive(APEX_FREQUENCIES_MHz, APEX_RAW_AMPLITUDES, APEX_PHASES_RAD, AMPLITUDE_CAP_MHz,
                       OLD_PEAK_CHECK_TIMES_US)
    scale_rad_per_us = 2 * np.pi * drive.amplitudes_MHz[0] / APEX_RAW_AMPLITUDES[0]
    assert np.isclose(scale_rad_per_us, APEX_SCALE, rtol=1e-10, atol=0)
    driven_yield, _ = yields(baseline_hypothesis(0.0), drive_rad_per_us(drive), APEX_TIMES_US,
                             OLD_COMPARISON_MAX_STEP_US)
    assert abs(driven_yield - 0.2041142705597517) < OLD_ENGINE_TOLERANCE


def test_apex_static_and_best_monochromatic_yields():
    # APEX δ = 0.0 static yield and best point of its monochromatic scan (28.12 MHz, amplitude exactly the cap,
    # phase 0, as APEX's mono_drive). From deltadelta_00_results.npz; may rely on stale placeholder parameters.
    # Measured differences: static about 3.4e-5 (the old engine's known static error), monochromatic about 2e-7.
    hypothesis = baseline_hypothesis(0.0)
    static_yield, _ = yields(hypothesis, None, APEX_TIMES_US, OLD_COMPARISON_MAX_STEP_US)
    assert abs(static_yield - 0.3844037855268709) < OLD_ENGINE_TOLERANCE
    monochromatic = Drive(np.array([28.12341137123746]), np.array([AMPLITUDE_CAP_MHz]), np.array([0.0]))
    monochromatic_yield, _ = yields(hypothesis, drive_rad_per_us(monochromatic), APEX_TIMES_US,
                                    OLD_COMPARISON_MAX_STEP_US)
    assert abs(monochromatic_yield - 0.25383613815719175) < OLD_ENGINE_TOLERANCE


# Seeds recorded in the old run logs: 1 MHz bins up to 40 MHz, 0.5 MHz minimum separation, 4 seeds.
# All DISCO/DISCO_run_1/run*/run*_run_log.txt and All APEX/APEX_Run_1/delta_*/*_run_log.txt.
# DISCO parameters, which may be stale placeholders. (DISCO run 1 = APEX δ 0.0 and DISCO run 2 = APEX δ 0.3.)
OLD_SEEDS_MHz = {
    0.0: [28.5, 27.5, 14.5, 13.5],
    0.1: [28.5, 27.5, 7.5, 30.5],
    0.2: [28.5, 27.5, 1.5, 0.5],
    0.3: [28.5, 27.5, 5.5, 33.5],
    0.4: [28.5, 27.5, 35.5, 11.5],
}
OLD_DISCRIMINATION_SEEDS_MHz = [28.5, 27.5, 5.5, 14.5]  # DISCO run 3: combined spectrum of δ 0.0 and 0.3


@pytest.mark.parametrize("delta", sorted(OLD_SEEDS_MHz))
def test_disruption_seeds_match_old_runs(delta):
    centres_MHz, scores = action_spectrum(baseline_hypothesis(delta), 1.0, 40.0)
    assert action_spectrum_seeds(centres_MHz, scores, 4, 0.5).tolist() == OLD_SEEDS_MHz[delta]


def test_discrimination_seeds_match_old_run3():
    # Discrimination seeds come from the element-wise maximum of the two scaled spectra, as in DISCO.
    centres_MHz, scores_A = action_spectrum(baseline_hypothesis(0.0), 1.0, 40.0)
    _, scores_B = action_spectrum(baseline_hypothesis(0.3), 1.0, 40.0)
    seeds = action_spectrum_seeds(centres_MHz, np.maximum(scores_A, scores_B), 4, 0.5)
    assert seeds.tolist() == OLD_DISCRIMINATION_SEEDS_MHz


# Starting scores of old optimiser runs: the seeds with raw amplitudes 1 and phases 0, L-infinity capped on the
# old 0–12 µs grid. history[0] in each run's results.npz. DISCO parameters, which may be stale placeholders.
def starting_drive(seeds_MHz):
    """The old optimiser's starting drive: raw amplitudes 1, phases 0, L-infinity cap on the old grid."""
    return linf_drive(seeds_MHz, np.ones(len(seeds_MHz)), np.zeros(len(seeds_MHz)), AMPLITUDE_CAP_MHz,
                      OLD_PEAK_CHECK_TIMES_US)


def test_starting_scores_match_old_runs():
    # Measured differences: about 1.5e-6 (run 1), 9e-7 (run 3), 2.4e-6 (APEX).
    times_us = np.linspace(0, 6, 2000)
    # DISCO run 1 (disrupt δ 0.0) and run 3 (discriminate δ 0.0 from 0.3), 0–6 µs.
    run1_cost = disruption_cost(baseline_hypothesis(0.0), starting_drive(OLD_SEEDS_MHz[0.0]), times_us,
                                OLD_COMPARISON_MAX_STEP_US)
    assert abs(run1_cost - 0.22797090857556898) < OLD_ENGINE_TOLERANCE
    run3_cost = discrimination_cost(baseline_hypothesis(0.0), baseline_hypothesis(0.3),
                                    starting_drive(OLD_DISCRIMINATION_SEEDS_MHz), times_us, OLD_COMPARISON_MAX_STEP_US)
    assert abs(run3_cost - (-0.012651528088392372)) < OLD_ENGINE_TOLERANCE
    # APEX δ 0.0 (disrupt), 0–10 µs with 3000 points.
    apex_cost = disruption_cost(baseline_hypothesis(0.0), starting_drive(OLD_SEEDS_MHz[0.0]), APEX_TIMES_US,
                                OLD_COMPARISON_MAX_STEP_US)
    assert abs(apex_cost - 0.2285173972315854) < OLD_ENGINE_TOLERANCE


def replaying_cost(kept_super_iterations):
    """Fake cost that makes the optimiser keep exactly the super-iterations an old run kept, and no others.

    It counts super-iterations by noticing when the frequencies change. A kept one scores lower than
    everything before it; any other scores 1, worse than anything. No physics is involved.
    """
    state = {"frequencies_MHz": None, "super_iteration": -1, "kept_so_far": 0}

    def cost(frequencies_MHz, raw_amplitudes, phases_rad):
        if not np.array_equal(frequencies_MHz, state["frequencies_MHz"]):
            state["frequencies_MHz"] = frequencies_MHz.copy()
            state["super_iteration"] += 1
            if state["super_iteration"] in kept_super_iterations:
                state["kept_so_far"] += 1
        if state["super_iteration"] == 0 or state["super_iteration"] in kept_super_iterations:
            return -state["kept_so_far"]
        return 1.0

    return cost


def test_replaying_old_choices_reproduces_old_frequencies():
    # Same seeds, same random nudges, same rounds kept => same final frequencies, whatever Nelder-Mead does.
    # Kept super-iterations from the full-precision histories in run3_results.npz (DISCO run 3, 20 rounds)
    # and deltadelta_00_results.npz (APEX δ 0.0, 14 rounds); random seed 123 in both. DISCO parameters,
    # which may be stale placeholders. DISCO run 1 kept only round 3 too and ended on APEX's frequencies.
    # Measured difference: about 5e-15 MHz (rounding only).
    run3 = optimise_drive(replaying_cost({1, 6, 7, 8, 9}), OLD_DISCRIMINATION_SEEDS_MHz, 20, 5.0, 5, 123, 20)
    assert np.allclose(run3.frequencies_MHz, RUN3_FREQUENCIES_MHz, rtol=0, atol=1e-9)
    apex = optimise_drive(replaying_cost({3}), OLD_SEEDS_MHz[0.0], 14, 5.0, 5, 123, 14)
    assert np.allclose(apex.frequencies_MHz, APEX_FREQUENCIES_MHz, rtol=0, atol=1e-9)


def test_patience_would_have_given_old_run3_result_sooner():
    # Old run 3 kept nothing after round 9, so with patience 6 it stops after round 15 with the same frequencies.
    run3 = optimise_drive(replaying_cost({1, 6, 7, 8, 9}), OLD_DISCRIMINATION_SEEDS_MHz, 20, 5.0, 5, 123, 6)
    assert len(run3.accepted) == 15
    assert np.allclose(run3.frequencies_MHz, RUN3_FREQUENCIES_MHz, rtol=0, atol=1e-9)


# FORGE N = 4 run with the power (L2) cap, from All FORGE/FORGE_Run_1/N4/N4_results.npz (frequencies converted to
# MHz). FORGE used DISCO's parameters, which may be stale placeholders, at δ 0.0 on 0–8 µs with 2000 points.
FORGE_TIMES_US = np.linspace(0, 8, 2000)
FORGE_L2_FREQUENCIES_MHz = [28.23187402054299, 23.750090105215392, 12.880640335452114, 16.824577810557706]
FORGE_L2_RAW_AMPLITUDES = [24.669561431527285, 29.724748537225103, 0.3426160595930887, 15.935752368314946]
FORGE_L2_PHASES_RAD = [2.0722414343990323, -3.823876087592403, 0.04224362583388336, 2.4535472085457366]
FORGE_L2_SCALE = 1.0525172792419843  # rad/µs per unit raw amplitude, as the old code stored it


def test_old_forge_power_capped_drive_gives_old_scale_and_yield():
    # Measured difference in the yield: about 9e-7. (All 12 FORGE runs, both caps, checked once on 2026-10-08:
    # largest difference 2e-5.)
    drive = l2_drive(FORGE_L2_FREQUENCIES_MHz, FORGE_L2_RAW_AMPLITUDES, FORGE_L2_PHASES_RAD, AMPLITUDE_CAP_MHz)
    scale_rad_per_us = 2 * np.pi * drive.amplitudes_MHz[0] / FORGE_L2_RAW_AMPLITUDES[0]
    assert np.isclose(scale_rad_per_us, FORGE_L2_SCALE, rtol=1e-10, atol=0)
    driven_yield = disruption_cost(baseline_hypothesis(0.0), drive, FORGE_TIMES_US, OLD_COMPARISON_MAX_STEP_US)
    assert abs(driven_yield - 0.22856499508084194) < OLD_ENGINE_TOLERANCE
