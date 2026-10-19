"""Small deterministic config: compute a fixed set of numbers with the code in one repo copy and save them.

    python behaviour_check.py <repo folder> <output .npz>

Run on the analysis copy once (baseline), then on the review copy after each batch of edits, and compare with
compare.py. Short time windows only, so it takes about a minute.
"""
import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo))

import control
import engine
import model
import results

out = {}


def put(name, value):
    out[name] = np.asarray(value)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", repo / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Settings for the check
TIMES_US = np.linspace(0, 0.3, 61)
MAX_STEP_US = 0.0016
SHORT_TIMES_US = np.linspace(0, 0.1, 21)
DIRECTIONS = (np.array([1.4393, 0.4242, 2.6949, 1.3668]), np.array([-2.6991, 3.0531, -0.5672, 3.127]))
A = model.flavin_tryptophan_hypothesis(0.0, 1.0, 1.0, 50.0)
B = model.flavin_tryptophan_hypothesis(np.pi / 2, 10.0, 1.0, 50.0)
CAP_MHz = 0.25 * A.zeeman_frequency_MHz
PEAK_TIMES_US = np.linspace(0, 12, 20000)
drive = control.linf_drive([3.98, 28.5, 40.5], [1.0, 0.6, 0.3], [0.2, 1.0, -0.5], CAP_MHz, PEAK_TIMES_US)
drive_function = control.drive_rad_per_us(drive)

# model
put("zeeman_MHz", model.zeeman_frequency_MHz(50.0))
put("angular", model.angular_frequency(np.array([1.0, 2.5])))
put("H_A", model.static_hamiltonian(A))
put("H_B", model.static_hamiltonian(B))
put("Heff_B", model.effective_hamiltonian(B))
put("K_B", model.recombination_operator(B))
put("P_B", model.recombination_projector(B))
put("X", model.drive_operator(A))
put("ciss", model.ciss_density_matrix(0.3))
put("singlet", model.singlet_density_matrix())
put("nuclear_dimension", model.nuclear_dimension(A))
put("R", model.rotation_matrix(0.7, 0.4))
put("U", model.spin_half_rotation(0.7, 0.4))
for label, turned in (("field", model.field_direction_hypothesis(B, 1.1, -2.3)),
                      ("oriented", model.oriented_hypothesis(B, 0.7, 0.4))):
    put(f"{label}_start", turned.electron_initial_state)
    put(f"{label}_recombination", turned.recombination_state)
    put(f"{label}_FN5", turned.nuclei[0].hyperfine_tensor_MHz)
    put(f"{label}_WN1", turned.nuclei[1].hyperfine_tensor_MHz)
    put(f"{label}_EED", turned.dipolar_tensor_MHz)
    put(f"{label}_H", model.static_hamiltonian(turned))

# engine
weights, kets = engine.initial_kets(B)
put("ket_weights", weights)
put("kets", kets)
X = model.drive_operator(A)
put("drive_exponential", engine.drive_exponential(X, X @ X, 0.37))
for index, (singlet, triplet) in enumerate(engine.populations_together([A, B], drive_function, TIMES_US, 2)):
    put(f"populations_{index}_S", singlet)
    put(f"populations_{index}_T", triplet)
put("yields_together", engine.yields_together([A, B], drive_function, TIMES_US, MAX_STEP_US))
put("yields_A_static", engine.yields(A, None, TIMES_US, MAX_STEP_US))
put("field_direction_yields_B", engine.field_direction_yields(B, drive_function, TIMES_US, MAX_STEP_US, DIRECTIONS))
put("recombination_yield_B", engine.recombination_yield(B, np.array([0.1, 0.2])))
put("contrast_A", engine.recombination_contrast(A, drive_function, TIMES_US, MAX_STEP_US, DIRECTIONS))
put("contrast_B", engine.recombination_contrast(B, drive_function, TIMES_US, MAX_STEP_US, DIRECTIONS))
grid = engine.orientation_grid(3)
put("grid", np.array(grid))
put("averaged_yields", engine.averaged_yields_together([A, B], drive_function, SHORT_TIMES_US, MAX_STEP_US, grid))
put("yields_averaged_route", engine.yields(B, drive_function, SHORT_TIMES_US, MAX_STEP_US, grid))
probabilities_A = engine.outcome_probabilities(0.3, 0.5)
probabilities_B = engine.outcome_probabilities(0.32, 0.47)
put("outcomes", probabilities_A)
put("bhattacharyya", engine.bhattacharyya_distance(probabilities_A, probabilities_B))
put("pairs_needed", engine.pairs_needed(engine.bhattacharyya_distance(probabilities_A, probabilities_B), 0.01))

# control
put("waveform", control.waveform(TIMES_US, [3.0, 7.0], [1.0, 0.5], [0.0, 1.0]))
put("linf_amplitudes", drive.amplitudes_MHz)
put("l2_amplitudes", control.l2_drive([3.98, 28.5], [1.0, 0.6], [0.2, 1.0], CAP_MHz).amplitudes_MHz)
put("drive_values", drive_function(TIMES_US))
turned_A = model.field_direction_hypothesis(A, 1.4393, -2.6991)
centres, scores = control.action_spectrum(turned_A, 1.0, 60.0)
put("action_centres", centres)
put("action_scores", scores)
put("action_scores_averaged", control.action_spectrum(A, 1.0, 60.0, grid)[1])
put("action_seeds", control.action_spectrum_seeds(centres, scores, 4, 2.0))
put("random_start", control.random_starting_frequencies(1, 4, 40.0))
put("disruption_cost", control.disruption_cost(A, drive, TIMES_US, MAX_STEP_US))
put("discrimination_cost", control.discrimination_cost(A, B, drive, TIMES_US, MAX_STEP_US))
put("bhattacharyya_cost", control.bhattacharyya_cost(A, B, drive, TIMES_US, MAX_STEP_US))
put("contrast_cost", control.contrast_cost(A, B, drive, TIMES_US, MAX_STEP_US, DIRECTIONS))


def small_cost(frequencies_MHz, raw_amplitudes, phases_rad):
    small_drive = control.linf_drive(frequencies_MHz, raw_amplitudes, phases_rad, CAP_MHz, PEAK_TIMES_US)
    return control.contrast_cost(A, B, small_drive, SHORT_TIMES_US, 0.0025, DIRECTIONS)


uninterrupted = control.optimise_drive(small_cost, [3.98, 28.5], 2, 5.0, 15, 1, 2)
for name in ("frequencies_MHz", "raw_amplitudes", "phases_rad", "cost_history", "accepted", "evaluations",
             "nelder_mead_costs"):
    put(f"optimise_{name}", getattr(uninterrupted, name))
with tempfile.TemporaryDirectory() as folder:
    checkpoint = Path(folder) / "random_seed_1.npz"
    control.optimise_drive(small_cost, [3.98, 28.5], 1, 5.0, 15, 1, 2, checkpoint_path=checkpoint)
    resumed = control.optimise_drive(small_cost, [3.98, 28.5], 2, 5.0, 15, 1, 2, checkpoint_path=checkpoint)
    put("resumed_cost_history", resumed.cost_history)
    put("resumed_nelder_mead_costs", resumed.nelder_mead_costs)

# results
with tempfile.TemporaryDirectory() as folder:
    path = results.save_run(folder, "check_2000-01-01_01", 7, {"times_us": TIMES_US, "goal": "discrimination",
                                                               "super_iterations": 2},
                            {"cost_history": np.array([-0.1, -0.2])})
    loaded = results.load_run(path)
    put("loaded_keys", sorted(loaded))
    put("loaded_times", loaded["times_us"])
    put("loaded_filled_in", loaded["filled_in"])
    put("same_settings", results.same_settings({"a": np.ones(3)}, {"a": np.ones(3)}))
put("durations", [results.format_duration(seconds) for seconds in (50, 500, 50000)])
put("progress", results.progress_line("super-iteration", 3, 20, 600.0))

# experiment scripts, with short windows patched into their settings
optimise_script = load_script("optimise_drive")
optimise_script.TIMES_US = TIMES_US
put("script_starting_frequencies", optimise_script.action_spectrum_starting_frequencies())
put("script_cost_contrast", optimise_script.cost(np.array([3.98, 28.5]), np.array([1.0, 0.5]), np.array([0.0, 1.0])))
put("script_both_contrasts", optimise_script.both_contrasts(drive))
put("script_both_contrasts_static", optimise_script.both_contrasts(None))
put("script_final_cost", optimise_script.final_cost((0.01,), (0.02,)))
put("script_make_drive", optimise_script.make_drive([3.98], [2.0], [0.1]).amplitudes_MHz)
optimise_script.DISCRIMINATION_MEASURE = "gap"
put("script_cost_gap", optimise_script.cost(np.array([3.98, 28.5]), np.array([1.0, 0.5]), np.array([0.0, 1.0])))
put("script_both_yields", optimise_script.both_yields(drive))
optimise_script.DISCRIMINATION_MEASURE = "bhattacharyya"
put("script_cost_bhattacharyya", optimise_script.cost(np.array([3.98]), np.array([1.0]), np.array([0.0])))

tone_script = load_script("single_tone_contrast")
tone_script.TIMES_US = TIMES_US
put("tone_yields_30MHz", tone_script.direction_yields(30.0))
put("tone_yields_static", tone_script.direction_yields(None))
put("tone_contrasts", tone_script.contrasts(tone_script.direction_yields(30.0)))

scan_script = load_script("field_scan")
scan_script.TIMES_US = TIMES_US
fn5_values, fn5_axes = np.linalg.eigh(scan_script.HYPOTHESIS_A.nuclei[0].hyperfine_tensor_MHz)
put("fn5_axes", fn5_axes)
put("unit_vector", scan_script.unit_vector(0.7, 0.4))
put("angles_to_axes", scan_script.angles_to_axes_deg(scan_script.unit_vector(0.7, 0.4), fn5_axes))
put("polished_extreme", scan_script.polished_extreme(scan_script.HYPOTHESIS_B, 1.3, 3.1, -1))
polar, azimuth = np.meshgrid(np.radians(np.arange(0, 181, 45.0)), np.radians(np.arange(-180, 180, 90.0)), indexing="ij")
scan_script.REFINE_EXTREMES = False
scan = scan_script.scan_one_pair(scan_script.HYPOTHESIS_B, (polar.ravel(), azimuth.ravel()), fn5_axes)
for name, value in scan.items():
    put(f"scan_{name}", value)

np.savez(output_path, **out)
print(f"Saved {len(out)} quantities to {output_path}")
