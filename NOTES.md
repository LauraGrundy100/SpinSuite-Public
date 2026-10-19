## What this is

The code I'm using to compare a classical and a CISS flavin-tryptophan radical pair and to optimise an RF drive that separates them by their compass contrast. 

    **Note to Daniel** -  This is a slightly trimmed copy of what I showed you on friday, containing the model, propagation, drive, scans, optimiser, plotting, results (from my thursday night run) for you to check my model etc.

    Apologies again for not explaining everything particularly well on Friday, I think that as I'd copied so much from the old code from last term, the scripts and commenting had made everything maybe *too* long to understand easily. I've also put together a model specification that I'll send alongside this!

## Files

Each function in the code has a tag saying where it came from:
`# old` = copied from last term's code unchanged; `# old, edited: ...` = started from last term's
code and then changed (the note says how); `# new` = written this term.
Anything not listed as carried over below is new.

- `model.py` (mostly new): spin operators, Hamiltonian, CISS start state, recombination projector and K,
  turning the molecule relative to the field, the flavin-tryptophan pair.
  - Carried over, unchanged: `drive_operator`.
  - Carried over, edited: `singlet_density_matrix` (NumPy instead of QuTiP), `spin_matrices` (any spin),
    `spin_operators` (any number of nuclei on either radical), `static_hamiltonian` (full tensors, nuclei on
    both radicals, dipolar term), `recombination_projector` (any recombination state, not only the singlet).
  - New: everything else, including `ciss_density_matrix`, `recombination_operator`, `effective_hamiltonian`,
    the rotation functions and `flavin_tryptophan_hypothesis`.
- `engine.py` (mostly new): propagation (Strang splitting plus Richardson extrapolation), yields, recombination
  yield, contrast, orientation averaging, Bhattacharyya distance.
  - Carried over, edited: `initial_kets` (splitting the start state into kets per nuclear state; now any mixed
    electron start, not only the singlet).
  - New: everything else.
- `control.py` (mostly carried over, edited): drive waveform, peak and power caps, action-spectrum starting
  frequencies, costs, the dCRAB-inspired optimiser with checkpoints.
  - Carried over, edited: `waveform`, `linf_drive`, `l2_drive`, `action_spectrum`, `action_spectrum_seeds`,
    `disruption_cost`, `discrimination_cost`, `one_super_iteration`, `optimise_drive`.
  - New: `drive_rad_per_us`, `random_starting_frequencies`, `bhattacharyya_cost`, `contrast_cost`,
    `new_search_state`, `save_search_state`, `load_search_state`.
- `results.py` (all new): run IDs, saving and loading runs, checkpoints, quick-look plots.
- `experiments/optimise_drive.py` (mostly new): the optimisation run.
  - Carried over, edited: `action_spectrum_starting_frequencies` (maximum also taken over the contrast directions).
  - New: everything else.
- `experiments/single_tone_contrast.py` (all new): contrast of both pairs under one sine at the cap, scanned over frequency.
- `experiments/field_scan.py` (all new): recombination yield over field directions in the molecule frame, no drive.
- `figures.py` (all new): the three figures in `figures/`.


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

## Thurday night's run

`discrimination_2026-10-15_01`: classical (A) against CISS (B), maximising |C_A − C_B| with C the highest minus lowest recombination yield over four field directions (the lowest and highest of each pair from `field_scan_2026-10-15_01`). 5 random seeds side by side, 5 rounds each, 4 frequencies from the action spectrum (up to 60 MHz), peak cap about 0.35 MHz, 0-12 µs on a 3 ns grid with 1.5 ns steps, Nelder-Mead capped at 1000 iterations per round. It ran for about 9 hours, roughly 18:00 to 03:00. Outputs: `results/discrimination_2026-10-15_01.npz` and `_cost.png`, and `figures/overnight_contrast.png`, which also uses the tone scan `single_tone_contrast_2026-10-15_01`. The field map comes from `field_scan_2026-10-15_02`.

## Not sure about

- How I read Daniel's k_r and k_f as K = k_r P_R + k_f·1, the CISS axis along molecule z (the EED axis), and electron 1 as the donor (`model.py`, `flavin_tryptophan_hypothesis`).
- The drive direction in the molecule frame, (cos θ cos φ, cos θ sin φ, −sin θ), is a convention. It matters because the molecule has no symmetry axis (`model.py`, `field_direction_hypothesis`).
- `oriented_hypothesis` only reaches field directions in the molecule's xz plane for this molecule. It's only used for orientation averaging, which is off (`model.py`).
- Why k_r is 10 for the CISS pair: Daniel's usual value, but I haven't written down the physical reason (`experiments/optimise_drive.py`).
- The contrast directions come from `field_scan_2026-10-15_01`, made by the earlier version of `field_scan.py` (590 Lebedev directions), which isn't in this copy. The map uses `_02`.
- `figures.py` hard-codes 5 rounds and 5 seeds in the overnight figure's labels.

## A note on AI usage
Keeping in line with the module guidance, I have used AI minimally in this project. All planning/notes etc. were done entirely by me, AI  has been used minimally for debugging, editing/clarifying comments, and similar. All ideas in this project are mine alone.