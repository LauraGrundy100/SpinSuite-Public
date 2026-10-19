import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# Each run is one results/<run ID>.npz holding its settings, outcome and a format version.
# Run IDs look like discrimination_2026-10-15_01.


# new
def new_run_id(results_folder, run_type):
    today = date.today().isoformat()
    run_number = 1
    while True:
        run_id = f"{run_type}_{today}_{run_number:02d}"
        if not ((Path(results_folder) / f"{run_id}.npz").exists()
                or checkpoint_folder(results_folder, run_id, finished=False).exists()
                or checkpoint_folder(results_folder, run_id, finished=True).exists()):
            return run_id
        run_number += 1


# new
def save_run(results_folder, run_id, format_version, settings, outcome):
    overlap = set(settings) & set(outcome)
    if overlap:
        raise ValueError(f"Names used in both settings and outcome: {sorted(overlap)}")
    path = Path(results_folder) / f"{run_id}.npz"
    if path.exists():
        raise FileExistsError(f"{path} already exists; get a fresh name from new_run_id")
    path.parent.mkdir(exist_ok=True)
    np.savez(path, format_version=format_version, run_id=run_id, **settings, **outcome)
    return path


# new
def load_run(path):
    return fill_known_old_settings(read_npz(path))


# new
def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].item() if data[name].ndim == 0 else data[name] for name in data.files}


# new
def fill_known_old_settings(run):
    # older format versions: rename old setting names and fill in settings they lacked; never touches results
    run["renamed"] = []
    for name in list(run):
        if name == "number_of_seeds" or name.endswith("seeds_MHz"):
            new_name = ("number_of_frequencies" if name == "number_of_seeds"
                        else name.removesuffix("seeds_MHz") + "starting_frequencies_MHz")
            run[new_name] = run.pop(name)
            run["renamed"].append(f"{name} -> {new_name}")
    run["filled_in"] = []
    is_optimise_drive_run = "goal" in run and "super_iterations" in run
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


# new
def checkpoint_folder(results_folder, run_id, finished):
    return Path(results_folder) / "checkpoints" / f"{run_id}_{'finished' if finished else 'unfinished'}"


# new
def save_checkpoint(path, **values):
    # write to a temporary file and swap it in, so an interruption cannot leave a half-written checkpoint
    path = Path(path)
    temporary_path = path.with_name(f"{path.stem}_writing.npz")
    with open(temporary_path, "wb") as file:
        np.savez(file, **values)
    temporary_path.replace(path)


# new
def same_settings(saved_settings, settings):
    return set(saved_settings) == set(settings) and all(
        np.array_equal(np.asarray(saved_settings[name]), np.asarray(settings[name])) for name in settings)


# new
def start_or_resume_run(results_folder, run_type, settings):
    # resume the newest unfinished run with exactly these settings, otherwise start a new one
    unfinished_folders = (Path(results_folder) / "checkpoints").glob(f"{run_type}_*_unfinished")
    for folder in sorted(unfinished_folders, reverse=True):
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


# new
def finish_run(results_folder, run_id, finished_runs_to_keep=3):
    # mark this run's checkpoints finished and delete all but the newest three finished runs' checkpoints
    checkpoint_folder(results_folder, run_id, finished=False).replace(
        checkpoint_folder(results_folder, run_id, finished=True))
    finished_folders = sorted((Path(results_folder) / "checkpoints").glob("*_finished"), key=last_written)
    for folder in finished_folders[:max(len(finished_folders) - finished_runs_to_keep, 0)]:
        shutil.rmtree(folder)


# new
def last_written(folder):
    return max([file.stat().st_mtime for file in folder.iterdir()] + [folder.stat().st_mtime])


# new
def save_quick_look(results_folder, run_id, plot_name, x_label, y_label, curves):
    # bare Figure so nothing opens a window; quick checks only, the proper figures come from figures.py
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


# new
def format_duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# new
def progress_line(label, done, total, elapsed_seconds):
    # time left assumes the remaining rounds take as long as the average so far
    seconds_left = elapsed_seconds / done * (total - done)
    return (f"{label} {done} of {total}, {format_duration(elapsed_seconds)} elapsed, "
            f"about {format_duration(seconds_left)} left")
