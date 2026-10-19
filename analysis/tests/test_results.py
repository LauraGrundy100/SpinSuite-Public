"""Tests for results.py. Files are written to pytest's temporary folder (tmp_path), never to results/."""

import os
from datetime import date

import numpy as np
import pytest

from results import (finish_run, format_duration, load_run, new_run_id, progress_line, read_npz, same_settings,
                     save_checkpoint, save_quick_look, save_run, start_or_resume_run)


# %% Run IDs, saving and loading

def test_run_ids_count_up_from_one(tmp_path):
    today = date.today().isoformat()
    assert new_run_id(tmp_path, "discrimination") == f"discrimination_{today}_01"
    (tmp_path / f"discrimination_{today}_01.npz").touch()
    assert new_run_id(tmp_path, "discrimination") == f"discrimination_{today}_02"
    assert new_run_id(tmp_path, "disruption") == f"disruption_{today}_01"  # each run type counts separately


def test_saved_run_loads_back_unchanged(tmp_path):
    settings = {"goal": "discrimination", "amplitude_cap_MHz": 7.0, "super_iterations": 20,
                "patience_super_iterations": 6, "discrimination_measure": "bhattacharyya",
                "times_us": np.linspace(0, 6, 5), "search_lebedev_order": 11, "final_lebedev_order": 17}
    outcome = {"cost_history": np.array([-0.1, -0.2]), "accepted": np.array([True])}
    save_run(tmp_path, "test_run", 3, settings, outcome)
    run = load_run(tmp_path / "test_run.npz")
    assert run["format_version"] == 3 and run["run_id"] == "test_run"
    assert run["goal"] == "discrimination" and run["amplitude_cap_MHz"] == 7.0 and run["super_iterations"] == 20
    assert np.array_equal(run["times_us"], settings["times_us"])
    assert np.array_equal(run["cost_history"], outcome["cost_history"])
    assert np.array_equal(run["accepted"], outcome["accepted"])
    assert run["filled_in"] == []  # a current file needs nothing filled in


def test_save_refuses_to_overwrite_or_mix_up_names(tmp_path):
    save_run(tmp_path, "test_run", 1, {"a": 1.0}, {"b": 2.0})
    with pytest.raises(FileExistsError):
        save_run(tmp_path, "test_run", 1, {"a": 1.0}, {"b": 2.0})
    with pytest.raises(ValueError):
        save_run(tmp_path, "other_run", 1, {"a": 1.0}, {"a": 2.0})


# %% Old optimise_drive.py files (format versions 1 and 2), written as the old save blocks did

def test_version_1_file_gets_patience_and_measure_filled_in(tmp_path):
    np.savez(tmp_path / "old.npz", format_version=1, run_id="old", goal="discrimination", super_iterations=20,
             cost_history=np.array([-0.02, -0.06]))
    run = load_run(tmp_path / "old.npz")
    assert run["patience_super_iterations"] == 20  # version 1 ran every super-iteration
    assert run["discrimination_measure"] == "gap"  # versions 1-2 always used the gap
    assert run["search_lebedev_order"] == 0 and run["final_lebedev_order"] == 0  # versions 1-4: no averaging
    assert run["filled_in"] == ["patience_super_iterations", "discrimination_measure", "search_lebedev_order", "final_lebedev_order"]
    assert "static_triplet_yield_A" not in run  # results the old file lacks are never invented


def test_version_2_file_gets_only_measure_filled_in(tmp_path):
    np.savez(tmp_path / "old.npz", format_version=2, run_id="old", goal="discrimination", super_iterations=20,
             patience_super_iterations=6)
    run = load_run(tmp_path / "old.npz")
    assert run["patience_super_iterations"] == 6
    assert run["filled_in"] == ["discrimination_measure", "search_lebedev_order", "final_lebedev_order"]


def test_old_names_for_starting_frequencies_are_renamed(tmp_path):
    # optimise_drive.py version 3 and disruption_sweep.py version 1 called the starting frequencies "seeds".
    np.savez(tmp_path / "old.npz", format_version=1, run_id="old", number_of_seeds=4, seeds_MHz=[28.5, 27.5],
             delta_0_3_seeds_MHz=[10.5, 20.5])
    run = load_run(tmp_path / "old.npz")
    assert run["number_of_frequencies"] == 4 and "number_of_seeds" not in run
    assert np.array_equal(run["starting_frequencies_MHz"], [28.5, 27.5]) and "seeds_MHz" not in run
    assert np.array_equal(run["delta_0_3_starting_frequencies_MHz"], [10.5, 20.5])
    assert run["renamed"] == ["number_of_seeds -> number_of_frequencies", "seeds_MHz -> starting_frequencies_MHz",
                              "delta_0_3_seeds_MHz -> delta_0_3_starting_frequencies_MHz"]


def test_null_test_file_loads_without_anything_filled_in(tmp_path):
    # A null-test file copies the optimised run's goal but has no search settings; it once made load_run crash.
    np.savez(tmp_path / "null.npz", format_version=1, run_id="null", goal="discrimination",
             discrimination_measure="bhattacharyya", number_beating=16, optimised_run_id="discrimination_x")
    run = load_run(tmp_path / "null.npz")
    assert run["number_beating"] == 16
    assert run["final_lebedev_order"] == 0 and run["filled_in"] == ["final_lebedev_order"]  # version 1: no averaging


def test_old_disruption_file_gets_no_discrimination_measure(tmp_path):
    np.savez(tmp_path / "old.npz", format_version=2, run_id="old", goal="disruption", super_iterations=14,
             patience_super_iterations=14)
    run = load_run(tmp_path / "old.npz")
    assert "discrimination_measure" not in run and run["filled_in"] == ["search_lebedev_order", "final_lebedev_order"]


def test_version_4_file_is_read_as_not_averaged(tmp_path):
    # Every run before 2026-10-13 used the molecules as given (no orientation averaging).
    np.savez(tmp_path / "old.npz", format_version=4, run_id="old", goal="discrimination", super_iterations=20,
             patience_super_iterations=6, discrimination_measure="bhattacharyya")
    run = load_run(tmp_path / "old.npz")
    assert run["search_lebedev_order"] == 0 and run["final_lebedev_order"] == 0
    assert run["filled_in"] == ["search_lebedev_order", "final_lebedev_order"]


# %% Checkpoints and resuming runs

SETTINGS = {"goal": "discrimination", "super_iterations": 20, "times_us": np.linspace(0, 6, 5)}


def test_checkpoint_is_written_whole_and_reads_back(tmp_path):
    save_checkpoint(tmp_path / "seed_1.npz", cost_history=[0.3, 0.2], position=7)
    save_checkpoint(tmp_path / "seed_1.npz", cost_history=[0.3, 0.2, 0.1], position=8)  # replaces the old one
    saved = read_npz(tmp_path / "seed_1.npz")
    assert np.array_equal(saved["cost_history"], [0.3, 0.2, 0.1]) and saved["position"] == 8
    assert [path.name for path in tmp_path.iterdir()] == ["seed_1.npz"]  # no temporary file left behind


def test_run_with_same_settings_is_resumed_and_changed_settings_start_afresh(tmp_path):
    today = date.today().isoformat()
    run_id, folder = start_or_resume_run(tmp_path, "discrimination", SETTINGS)
    assert run_id == f"discrimination_{today}_01" and folder.name == f"{run_id}_unfinished"
    assert same_settings(read_npz(folder / "settings.npz"), SETTINGS)
    assert start_or_resume_run(tmp_path, "discrimination", SETTINGS) == (run_id, folder)  # resumed
    changed = dict(SETTINGS, times_us=np.linspace(0, 6, 6))
    assert start_or_resume_run(tmp_path, "discrimination", changed)[0] == f"discrimination_{today}_02"
    assert start_or_resume_run(tmp_path, "disruption", SETTINGS)[0] == f"disruption_{today}_01"  # other types apart


def test_finished_run_is_labelled_and_not_resumed(tmp_path):
    run_id, _ = start_or_resume_run(tmp_path, "discrimination", SETTINGS)
    finish_run(tmp_path, run_id)
    assert (tmp_path / "checkpoints" / f"{run_id}_finished" / "settings.npz").exists()
    assert start_or_resume_run(tmp_path, "discrimination", SETTINGS)[0] != run_id  # same settings, new run


def test_only_the_newest_three_finished_runs_keep_checkpoints(tmp_path):
    unfinished_id, _ = start_or_resume_run(tmp_path, "disruption", SETTINGS)  # never deleted, however old
    finished_ids = []
    for number in range(4):
        run_id, folder = start_or_resume_run(tmp_path, "discrimination", dict(SETTINGS, super_iterations=number))
        for path in (folder / "settings.npz", folder):  # pretend run k last wrote on day k, so run 0 is oldest
            os.utime(path, (1e9 + number * 86400, 1e9 + number * 86400))
        finish_run(tmp_path, run_id)
        finished_ids.append(run_id)
    remaining = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
    assert remaining == sorted([f"{unfinished_id}_unfinished"] + [f"{run_id}_finished" for run_id in finished_ids[1:]])


# %% Quick-look plots

def test_quick_look_is_saved_as_png_named_by_run_id(tmp_path):
    curves = [(np.arange(3), np.array([0.3, 0.2, 0.2]), "delta = 0.0"), (np.arange(2), np.array([0.4, 0.3]), "delta = 0.3")]
    path = save_quick_look(tmp_path, "test_run", "cost", "super-iteration", "cost", curves)
    assert path == tmp_path / "test_run_cost.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # every PNG file starts with these 8 bytes


# %% Progress printing

def test_durations_switch_units_at_two_minutes_and_two_hours():
    assert format_duration(45) == "45 s"
    assert format_duration(600) == "10.0 min"
    assert format_duration(9000) == "2.5 h"


def test_progress_line_worked_example():
    # 3 of 20 done in 600 s: 200 s each on average, so 17 left take 3400 s = 56.7 min.
    line = progress_line("super-iteration", 3, 20, 600.0)
    assert line == "super-iteration 3 of 20, 10.0 min elapsed, about 56.7 min left"
    assert line.isascii()  # non-ASCII text once crashed printing to a log on Windows
