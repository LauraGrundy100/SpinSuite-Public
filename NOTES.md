## What this is

The code I'm using to compare a classical and a CISS flavin-tryptophan radical pair and to optimise an RF drive that separates them by their compass contrast. It's a trimmed copy with just the model, propagation, drive, scans, optimiser, plotting, and the results from the overnight run.

## Files

- `model.py`: spin operators, Hamiltonian, CISS start state, recombination projector and K, turning the molecule relative to the field, the flavin-tryptophan pair. New; spin operators, Hamiltonian and singlet projector started from last year's code.
- `engine.py`: propagation (Strang splitting plus Richardson extrapolation), yields, recombination yield, contrast, orientation averaging, Bhattacharyya distance. New; splitting the start state into kets per nuclear state is last year's, edited.
- `control.py`: drive waveform, peak and power caps, action-spectrum starting frequencies, costs, the dCRAB-style optimiser with checkpoints. New; the waveform, caps, action spectrum and optimiser loop are last year's, edited.
- `results.py`: run IDs, saving and loading runs, checkpoints, quick-look plots. New.
- `experiments/optimise_drive.py`: the optimisation run. New.
- `experiments/single_tone_contrast.py`: contrast of both pairs under one sine at the cap, scanned over frequency. New.
- `experiments/field_scan.py`: recombination yield over field directions in the molecule frame, no drive. New.
- `figures.py`: the three figures in `figures/`. New.

Each function has a tag: `# old`, `# old, edited: ...` or `# new`.

## How to run

From the repo folder, in a terminal (the scripts with several processes have to be started from the terminal, not cell by cell):

- single-frequency scan: `python experiments/single_tone_contrast.py`
- optimisation: `python experiments/optimise_drive.py`. If it's interrupted, starting it again with the same settings carries on from `results/checkpoints/`.
- field scan for the map: `python experiments/field_scan.py`
- figures: `python figures.py`. It reads the three runs in `results/` and rewrites `figures/*.png` (plus PDFs).

Settings are the block at the top of each script. I used numpy, scipy (needs `scipy.integrate.lebedev_rule`), qutip and matplotlib; the PNGs were made with matplotlib 3.11.1.

## Conventions

- Frame: FN5, WN1 and EED are given in the molecule frame. The static field is always along lab z and the drive along lab x. To put the field along a direction (θ, φ) of the molecule I turn the molecule instead (`field_direction_hypothesis`): tensors, CISS start state and recombination projector all turn together. Exchange is zero, so the dipolar term is the only coupling between the radicals.
- Units: settings in MHz, µs and µs⁻¹; matrices in rad/µs. The only MHz to rad/µs conversions are `angular_frequency` (model) and `drive_rad_per_us` (drive amplitudes). Rates never get a 2π.
- Parameters: tensors and J = 0 in `flavin_tryptophan_hypothesis` (`model.py`). Field strength, χ, k_r and k_f, the cap, the time window, the step and the contrast directions are in each script's settings block, copied by hand between the three scripts.
- From Daniel: the three tensors, k_r = k_f = 1 µs⁻¹ (classical), χ = π/2 with k_r = 10, k_f = 1 µs⁻¹ (CISS), the form of the CISS start and recombination states, neglecting exchange, a 10-12 µs window, and the contrast as the measure. My choices: 50 µT, cap 0.25 × Zeeman (about 0.35 MHz, placeholder), the drive direction, the CISS axis along molecule z.
- K and rates: K = k_S P_R + k_T (1 − P_R) with k_S = k_r + k_f and k_T = k_f, so K = k_r P_R + k_f·1. It enters through H − iK/2 in `effective_hamiltonian`. The recombination yield keeps the k_r share of k_S ∫ Tr(P_R ρ) dt.

## Last night's run

`discrimination_2026-10-15_01`: classical (A) against CISS (B), maximising |C_A − C_B| with C the highest minus lowest recombination yield over four field directions (the lowest and highest of each pair from `field_scan_2026-10-15_01`). 5 random seeds side by side, 5 rounds each, 4 frequencies from the action spectrum (up to 60 MHz), peak cap about 0.35 MHz, 0-12 µs on a 3 ns grid with 1.5 ns steps, Nelder-Mead capped at 1000 iterations per round. It ran for about 9 hours, roughly 18:00 to 03:00. Outputs: `results/discrimination_2026-10-15_01.npz` and `_cost.png`, and `figures/overnight_contrast.png`, which also uses the tone scan `single_tone_contrast_2026-10-15_01`. The field map comes from `field_scan_2026-10-15_02`.

## Not sure about

- How I read Daniel's k_r and k_f as K = k_r P_R + k_f·1, the CISS axis along molecule z (the EED axis), and electron 1 as the donor (`model.py`, `flavin_tryptophan_hypothesis`).
- The drive direction in the molecule frame, (cos θ cos φ, cos θ sin φ, −sin θ), is a convention. It matters because the molecule has no symmetry axis (`model.py`, `field_direction_hypothesis`).
- `oriented_hypothesis` only reaches field directions in the molecule's xz plane for this molecule. It's only used for orientation averaging, which is off (`model.py`).
- Why k_r is 10 for the CISS pair: Daniel's usual value, but I haven't written down the physical reason (`experiments/optimise_drive.py`).
- The contrast directions come from `field_scan_2026-10-15_01`, made by the earlier version of `field_scan.py` (590 Lebedev directions), which isn't in this copy. The map uses `_02`.
- `figures.py` hard-codes 5 rounds and 5 seeds in the overnight figure's labels.

## Supervision meeting

- Check that the CISS initial state and recombination projectors stay consistent with the tensors under rotation.
- Try the drive along the static field.
- Minimise the classical-model contrast first, then apply that drive to the CISS model.
- Look into the classical-model behaviour in the run that I couldn't explain.
- Write up the model specification.
