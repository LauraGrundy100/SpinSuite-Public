"""Check that a repo copy's figures.py still reads the results and draws the same figures.

    python figure_check.py make-proxy <repo> <results folder>    small stand-in runs under the real run IDs
    python figure_check.py draw <repo> <results folder> <out>    run <repo>/figures.py on those results; PNGs in <out>/figures
    python figure_check.py compare <png folder 1> <png folder 2> pixel-by-pixel comparison of every PNG

figures.py is run with its folder redirected to <out>, so it never overwrites the figures kept in the repo.
"""
import os

os.environ["MPLBACKEND"] = "Agg"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

mode = sys.argv[1]


def load_script(repo, name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", repo / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if mode == "make-proxy":
    repo, folder = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
    sys.path.insert(0, str(repo))
    from results import save_run

    times_us = np.linspace(0, 0.5, 101)
    scan = load_script(repo, "field_scan")
    scan.TIMES_US, scan.REFINE_EXTREMES = times_us, False
    _, fn5_axes = np.linalg.eigh(scan.HYPOTHESIS_A.nuclei[0].hyperfine_tensor_MHz)
    map_polar = np.radians(np.arange(0.0, 180.0 + 1e-9, 30.0))
    map_azimuth = np.radians(np.arange(-180.0, 180.0 - 1e-9, 30.0))
    polar, azimuth = np.meshgrid(map_polar, map_azimuth, indexing="ij")
    outcome = {"fn5_axes": fn5_axes, "map_polar_angles_rad": map_polar, "map_azimuths_rad": map_azimuth}
    for label, hypothesis in (("A", scan.HYPOTHESIS_A), ("B", scan.HYPOTHESIS_B)):
        pair = scan.scan_one_pair(hypothesis, (polar.ravel(), azimuth.ravel()), fn5_axes)
        outcome.update({f"{name}_{label}": value for name, value in pair.items()})
    save_run(folder, "field_scan_2026-10-15_02", 2, {"proxy": "yes"}, outcome)

    tones = load_script(repo, "single_tone_contrast")
    tones.TIMES_US = times_us
    frequencies = np.array([4.0, 30.0, 32.5, 40.5])
    static = tones.contrasts(tones.direction_yields(None))
    tone_contrasts = np.array([tones.contrasts(tones.direction_yields(f)) for f in frequencies])
    save_run(folder, "single_tone_contrast_2026-10-15_01", 1,
             {"amplitude_cap_MHz": tones.AMPLITUDE_CAP_MHz, "tone_frequencies_MHz": frequencies},
             {"static_contrast_A": static[0], "static_contrast_B": static[1],
              "tone_contrasts_A": tone_contrasts[:, 0], "tone_contrasts_B": tone_contrasts[:, 1]})

    optimise = load_script(repo, "optimise_drive")
    optimise.TIMES_US, optimise.SUPER_ITERATIONS, optimise.PATIENCE_SUPER_ITERATIONS = times_us, 2, 2
    optimise.MAX_NELDER_MEAD_ITERATIONS, optimise.NUMBER_OF_FREQUENCIES, optimise.RANDOM_SEEDS = 10, 2, [1, 2]
    with tempfile.TemporaryDirectory() as checkpoints:
        outcomes = {seed: optimise.run_one_seed(seed, Path(checkpoints)) for seed in optimise.RANDOM_SEEDS}
    static_outcome = optimise.print_contrast_summary(outcomes)
    best_seed = min(optimise.RANDOM_SEEDS, key=lambda seed: outcomes[seed]["final_cost"])
    outcome = dict(**static_outcome, best_random_seed=best_seed, **outcomes[best_seed],
                   **{f"random_seed_{seed}_{name}": value for seed in optimise.RANDOM_SEEDS
                      for name, value in outcomes[seed].items()})
    save_run(folder, "discrimination_2026-10-15_01", 7,
             {"random_seeds": optimise.RANDOM_SEEDS, "amplitude_cap_MHz": optimise.AMPLITUDE_CAP_MHz}, outcome)
    print(f"Proxy runs saved in {folder}")

elif mode == "draw":
    repo, results_folder, out = (Path(argument).resolve() for argument in sys.argv[2:5])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results").exists():
        shutil.rmtree(out / "results")
    shutil.copytree(results_folder, out / "results")
    sys.path.insert(0, str(repo))
    source = (repo / "figures.py").read_text(encoding="utf-8")
    exec(compile(source, str(repo / "figures.py"), "exec"), {"__file__": str(out / "figures.py"), "__name__": "__main__"})

elif mode == "compare":
    import matplotlib.image

    first, second = Path(sys.argv[2]), Path(sys.argv[3])
    names = sorted(path.name for path in first.glob("*.png"))
    failures = []
    for name in names:
        if not (second / name).exists():
            failures.append(f"{name}: missing")
            continue
        a, b = matplotlib.image.imread(first / name), matplotlib.image.imread(second / name)
        if a.shape != b.shape or not np.array_equal(a, b):
            failures.append(f"{name}: pixels differ" if a.shape == b.shape else f"{name}: size {a.shape} vs {b.shape}")
    print(f"{len(names)} figures compared: {names}")
    print("PASS" if names and not failures else "FAIL\n  " + "\n  ".join(failures or ["no figures"]))
    sys.exit(1 if failures or not names else 0)
