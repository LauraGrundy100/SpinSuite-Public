"""Run records: run IDs, saving and loading results with their settings, checkpoints, quick-look plots,
progress printing.

Every run is saved as one .npz file, results/<run ID>.npz, holding its run ID, a format version, the settings
and the outcome, plus plain quick-look PNGs results/<run ID>_<plot name>.png.
A run ID names what the run did and when, e.g. "discrimination_2026-10-09_01".
While a run is going, its settings and checkpoints sit in results/checkpoints/<run ID>_unfinished/; the folder is
renamed <run ID>_finished/ once the run is saved, and only the newest three finished runs' folders are kept.
Library file: no settings and no module-level state. Printed text is plain ASCII (Windows logs).
"""

import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def new_run_id(results_folder, run_type):
    """The next free run ID for today: "<run_type>_<YYYY-MM-DD>_<nn>", nn counting up from 01.

    An ID is taken if it has a results file or a checkpoint folder (finished or not).
    """
    today = date.today().isoformat()
    run_number = 1
    while True:
        run_id = f"{run_type}_{today}_{run_number:02d}"
        if not ((Path(results_folder) / f"{run_id}.npz").exists()
                or checkpoint_folder(results_folder, run_id, finished=False).exists()
                or checkpoint_folder(results_folder, run_id, finished=True).exists()):
            return run_id
        run_number += 1


def save_run(results_folder, run_id, format_version, settings, outcome):
    """Save a run as results_folder/<run_id>.npz with its run ID, format version, settings and outcome.

    settings and outcome are dicts of name -> number, string or array; their names must not overlap.
    Refuses to overwrite an existing file. Returns the path saved to.
    """
    overlap = set(settings) & set(outcome)
    if overlap:
        raise ValueError(f"Names used in both settings and outcome: {sorted(overlap)}")
    path = Path(results_folder) / f"{run_id}.npz"
    if path.exists():
        raise FileExistsError(f"{path} already exists; get a fresh name from new_run_id")
    path.parent.mkdir(exist_ok=True)
    np.savez(path, format_version=format_version, run_id=run_id, **settings, **outcome)
    return path


def load_run(path):
    """Load a saved run as a plain dict. Single values come back as plain numbers or strings, arrays as arrays.

    Settings that older format versions did not store, but whose meaning is known, are filled in
    (see fill_known_old_settings) and their names listed under "filled_in".
    """
    return fill_known_old_settings(read_npz(path))


def read_npz(path):
    """Read an .npz file as a plain dict, single values as plain numbers or strings. No pickled objects."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].item() if data[name].ndim == 0 else data[name] for name in data.files}


def fill_known_old_settings(run):
    """Bring old files up to today's names. Never adds results.

    Renames (listed under "renamed"): number_of_seeds -> number_of_frequencies, and any name ending seeds_MHz ->
    ending starting_frequencies_MHz (optimise_drive.py up to version 3, disruption_sweep.py version 1).
    Fills in (listed under "filled_in") settings old optimise_drive.py files lacked, with the value the old code
    used: version 1 had no patience setting (every super-iteration ran, so patience = super_iterations);
    versions 1-2 had no discrimination measure (discrimination always used the gap); versions 1-4 had no
    orientation averaging (search_lebedev_order = final_lebedev_order = 0, the molecules as given); null_test.py
    version 1 had no orientation averaging either (final_lebedev_order = 0).
    """
    run["renamed"] = []
    for name in list(run):
        if name == "number_of_seeds" or name.endswith("seeds_MHz"):
            new_name = ("number_of_frequencies" if name == "number_of_seeds"
                        else name.removesuffix("seeds_MHz") + "starting_frequencies_MHz")
            run[new_name] = run.pop(name)
            run["renamed"].append(f"{name} -> {new_name}")
    run["filled_in"] = []
    is_optimise_drive_run = "goal" in run and "super_iterations" in run  # null-test files store a goal but no search
    if is_optimise_drive_run and "patience_super_iterations" not in run:
        run["patience_super_iterations"] = run["super_iterations"]
        run["filled_in"].append("patience_super_iterations")
    if is_optimise_drive_run and run["goal"] == "discrimination" and "discrimination_measure" not in run:
        run["discrimination_measure"] = "gap"
        run["filled_in"].append("discrimination_measure")
    for name in ("search_lebedev_order", "final_lebedev_order"):
        if is_optimise_drive_run and name not in run:
            run[name] = 0
            run["filled_in"].append(name)
    is_null_test_run = "optimised_run_id" in run
    if is_null_test_run and "final_lebedev_order" not in run:
        run["final_lebedev_order"] = 0
        run["filled_in"].append("final_lebedev_order")
    return run


# %% Checkpoints: results/checkpoints/<run ID>_unfinished/ (renamed <run ID>_finished/ when the run is saved)

def checkpoint_folder(results_folder, run_id, finished):
    """The folder holding a run's settings.npz and checkpoint files, labelled finished or unfinished."""
    return Path(results_folder) / "checkpoints" / f"{run_id}_{'finished' if finished else 'unfinished'}"


def save_checkpoint(path, **values):
    """Write values to path (.npz) safely: first to a temporary file, then swapped into place in one step,
    so an interruption while writing leaves the previous checkpoint intact."""
    path = Path(path)
    temporary_path = path.with_name(f"{path.stem}_writing.npz")
    with open(temporary_path, "wb") as file:
        np.savez(file, **values)
    temporary_path.replace(path)


def same_settings(saved_settings, settings):
    """True if both dicts have the same names and equal values (arrays compared element by element, exactly)."""
    return set(saved_settings) == set(settings) and all(
        np.array_equal(np.asarray(saved_settings[name]), np.asarray(settings[name])) for name in settings)


def start_or_resume_run(results_folder, run_type, settings):
    """Run ID and checkpoint folder for a run with these settings.

    Resumes the newest unfinished run of this type whose saved settings are exactly these; otherwise starts
    a new run, making its checkpoint folder and saving the settings there.
    """
    unfinished_folders = (Path(results_folder) / "checkpoints").glob(f"{run_type}_*_unfinished")
    for folder in sorted(unfinished_folders, reverse=True):  # newest date and number first
        run_id = folder.name.removesuffix("_unfinished")
        is_this_type = re.fullmatch(rf"{re.escape(run_type)}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}", run_id)
        if (is_this_type and (folder / "settings.npz").exists()
                and same_settings(read_npz(folder / "settings.npz"), settings)):
            print(f"Resuming unfinished run {run_id} (same settings)", flush=True)
            return run_id, folder
    run_id = new_run_id(results_folder, run_type)
    folder = checkpoint_folder(results_folder, run_id, finished=False)
    folder.mkdir(parents=True)
    np.savez(folder / "settings.npz", **settings)
    print(f"Starting run {run_id}", flush=True)
    return run_id, folder


def finish_run(results_folder, run_id, finished_runs_to_keep=3):
    """Call after save_run. Labels the run's checkpoint folder finished, then deletes the checkpoints of older
    finished runs, keeping the newest finished_runs_to_keep. Unfinished runs are never touched."""
    checkpoint_folder(results_folder, run_id, finished=False).replace(
        checkpoint_folder(results_folder, run_id, finished=True))
    finished_folders = sorted((Path(results_folder) / "checkpoints").glob("*_finished"), key=last_written)
    for folder in finished_folders[:max(len(finished_folders) - finished_runs_to_keep, 0)]:
        shutil.rmtree(folder)


def last_written(folder):
    """When anything in the folder was last written (s since 1970), i.e. roughly when that run finished."""
    return max([file.stat().st_mtime for file in folder.iterdir()] + [folder.stat().st_mtime])


# %% Quick-look plots

def save_quick_look(results_folder, run_id, plot_name, x_label, y_label, curves):
    """Save a plain line plot as results_folder/<run_id>_<plot_name>.png. Never used in reports (see figures.py).

    curves is a list of (x, y, label). Drawn on a bare Figure, which writes straight to the file without
    opening a window, so it works the same in the terminal and the interactive window. Returns the path.
    """
    figure = Figure(figsize=(6, 4))
    axes = figure.add_subplot()
    for x, y, label in curves:
        axes.plot(x, y, marker=".", label=label)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(run_id)
    if len(curves) > 1:
        axes.legend()
    figure.tight_layout()
    path = Path(results_folder) / f"{run_id}_{plot_name}.png"
    figure.savefig(path, dpi=100)
    return path


# %% Progress printing

def format_duration(seconds):
    """A duration as plain text: seconds below 2 minutes, minutes below 2 hours, then hours."""
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def progress_line(label, done, total, elapsed_seconds):
    """E.g. "super-iteration 3 of 20, 10.0 min elapsed, about 56.7 min left".

    Time left assumes each remaining step takes the average time of the steps done so far, so it is rough
    when steps vary in length; an optimiser stopping early (patience) can finish sooner.
    """
    seconds_left = elapsed_seconds / done * (total - done)
    return (f"{label} {done} of {total}, {format_duration(elapsed_seconds)} elapsed, "
            f"about {format_duration(seconds_left)} left")
