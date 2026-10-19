# CLAUDE.md

## Read this first

Laura Grundy is a final-year MSci Natural Sciences (physics) student at the University of Exeter. This repo is her dissertation code. She will work on it almost every day for about eight months, and she must be able to explain any part of it to her supervisor, Dr Daniel Kattnig, in a meeting with no preparation.

Why this repo exists: last term's code became a tangle. Many scripts each held a copy of the same 150 lines, settings were buried in globals, runs took 30+ hours, and some parts were added quickly with AI help that Laura did not fully understand. That made her life hard when she was asked questions about her own work. Everything below follows from one aim: **Laura can read, explain, change and trust every line.**

When goals pull against each other, use this order:

1. Correct (results are right, and you have actually run them)
2. Understandable by Laura
3. Simple
4. Fast
5. Extensible (only for things already planned, never for imagined futures)

Speed matters because of the long runtimes, but never buy speed with something Laura cannot explain.

---
## Explaining things to Laura (applies to every reply, every milestone)

Laura is a capable physicist and programmer, but she has told us she is losing track of what is happening in the repo. She must be able to follow every reply without having read the code. This section overrides any habit of writing dense technical summaries.

1. **Open with a plain-language summary of two to four sentences.** Say what you did or propose and why, in everyday words. No file names, function names or jargon in this opening paragraph.
2. **Define every technical term the first time it appears in a reply.** This includes numerical methods (Strang splitting, Richardson extrapolation, Nelder-Mead), statistics (Fisher information, QFI) and software terms (dataclass, regression test). Give one sentence of definition plus an everyday analogy or a small worked example with numbers.
3. **Say what it means for her.** What will she be able to do or explain afterwards? What does she have to decide, and what is the default if she has no view?
4. **Give a three-line map at the start and end of every step.** What exists now, what comes next, what is waiting on her.
5. **Apply the meeting test.** If Daniel asked "what did you just do, and why?", could Laura answer from your first paragraph alone? If not, rewrite it.
6. **Keep replies short.** Put detail behind an offer ("I can go through the maths of this if you want") instead of including it all. No tables of settings or long lists of numbers unless she asks.
7. **A go-ahead only counts if she has seen the plain-language version.** Never treat silence, or approval of a technical description she did not follow, as permission to build.
8. **If she says she is lost, stop building.** Re-explain from the top of the project in plain language (what the repo does, what exists, what is next) and wait.
9. **Say what you cannot see or have not run.** If you did not run something or cannot see a file, say so in the first paragraph.

## The six rules

### 1. Keep it simple

Complexity is the main risk in this repo. It is easier to add a clever abstraction than to remove one later, and Laura has to live with whatever is added.

- Use plain functions and dataclasses. Do not add base classes, registries, plugin systems, decorators, metaclasses, config frameworks (YAML, JSON, hydra) or dependency injection unless Laura asks for them.
- Build for the current milestone only. Leave room to grow by keeping things small and clearly named, not by building machinery in advance. The exception is a later *planned* goal that is cheaper to include now than to come back for (see "Build order").
- A function longer than about 40 lines, or taking more than about 5 arguments, is a signal to stop and discuss simplifying or splitting it.
- Test for every piece of code: could Laura explain this to Daniel after five minutes of reading? If not, simplify it.

### 2. Understanding comes before implementation

- Before adding any new concept (an algorithm, a numerical trick, a data structure, a library), explain it in plain language first: what it is, why it is needed here, what it replaces, and how Laura can check that it works. Then wait for her go-ahead.
- If you cannot explain it simply, do not add it.
- After implementing, give a short note: what changed, where, and how to check it.
- Every function gets a docstring of one or two lines: what it does, the units, and any key assumption.
- Do not add optimisations Laura has not approved. Where a fast version of something exists, keep a slow, readable reference version in `tests/` and test that the two agree. That way Laura can always show Daniel the transparent version.

### 3. Few files

The fewer things Laura has to switch between, the better. Use the layout in "Planned structure" below.

- Do not create new files without asking.
- Split a file only when it passes a few hundred lines, or when a second file needs to import part of it.
- Plumbing (saving, checkpointing, progress printing) lives in `results.py` so that experiment scripts stay readable and contain only what Laura actually touches.

### 4. Settings at the top, working code below

- Every experiment script starts with one settings block: everything Laura might want to change, with clear names and units in comments. Then a row or two of `####`, then the working code.
- Use `# %%` cell separators so scripts run either cell by cell in the VS Code interactive window or as a plain script from the terminal.
- This layout is only a convenience. If following it would make a script harder to understand, do not follow it, and tell Laura why.
- Library files (`model.py`, `engine.py`, `control.py`, `results.py`) have no settings and no module-level state. Values are passed in as arguments, so changing something in one place can never silently change another place. (Last term, a global `tlist` was shared by everything.)

### 5. Easy to find things

Laura may be asked "where is X?" in a meeting with the whole repo open.

- Use full-word names with units: `hyperfine_coupling_MHz`, `drive_frequency_MHz`, `singlet_yield`. Avoid single letters such as `A`, `w`, `k`. Standard physics symbols (for example `delta` for the hyperfine anisotropy parameter) are fine if explained in a docstring where they first appear.
- `README.md` has a contents section (file, then what it holds), an experiment index, a figure index, and a "where is X?" section answering likely supervisor questions: where is the Hamiltonian, how are yields computed, where is the drive normalised, and so on.
- Whenever you add, move, rename or delete something, update the README in the same step.

### 6. Do not break things

- Change existing function signatures and saved-file formats only deliberately. Before changing a function that is used elsewhere, search for every use and tell Laura what will be affected.
- Every saved result stores its settings and a format version, so old results stay interpretable after the code changes.
- Run the tests after every change.
- Port working code from the old repo. Laura wants to keep what she likes and leave the rest behind.

---

## When a rule is wrong for the situation, or Laura is wrong

Laura is sometimes wrong, and these rules are not sacred. If you see a better approach that would break a rule, or you think a decision of hers is a mistake, say so. Do not silently comply, and do not silently override. Stop and ask, using this structure:

1. **What you want to do**, in plain language.
2. **Why**, with the concrete benefit.
3. **Which rule or decision it conflicts with.**
4. **What happens if we keep the rule**: the real cost or implication of sticking to it.
5. **Your recommendation**, and any alternatives.

Then wait. Laura decides.

The same standard applies to every question you ask. Explain what you want to do and why before asking. Explain any jargon. Offer a default. Do not ask questions you could answer by reading the repo.

---

## Working method

- **One small step at a time.** One function or one module, then stop. Show what you did and wait for review before the next step.
- **Never claim without running.** Do not state a result, a timing, or that something works unless you ran it. If you did not run it, say so plainly.
- **Review means no edits.** When Laura asks you to review, audit or explain code, do not change files. Report findings and ask.
- **Commit after each approved step**, with a message saying what changed and why. Do not commit `results/` data or figures (keep them in `.gitignore`).
- **Long jobs:** say roughly how long something will take before running it. Do not run anything longer than two or three minutes without telling Laura first.
- **Missing files:** if a file the conversation refers to is not there, say so. Do not proceed on assumptions.
- **The old repo** (`LauraGrundy100/Spin-Project`, called "the old repo") is read-only reference. Never modify it. Port code by copying with minimal changes, after explaining.

---

## The project in brief

The physics: a radical pair is two electron spins (one on each radical) coupled to nuclear spins. After the pair forms in a spin state (currently a singlet), the spins evolve under a Hamiltonian and the pair recombines. The fraction that recombines through the singlet channel is the **singlet yield**. This is the model for the proposed cryptochrome compass in migratory birds.

Laura applies an optimised radiofrequency (RF) drive to the pair. The old experiments were:

- **Disruption**: find a drive that suppresses the singlet yield (APEX, FORGE).
- **Discrimination**: find a drive that maximises the gap `D = |Y_A − Y_B|` between the yields of two competing Hamiltonians (DISCO).

APEX, DISCO and FORGE were separate only because each new idea got its own set of scripts as the project grew. That split does not carry over. The new repo combines the best parts of all three in shared code (for example, one optimiser with both L-infinity and L2 normalisation), and experiments are organised by what they do, not by which old script they came from.

This term's direction: discriminate between competing hypotheses about the radical pair (for example singlet-born vs CISS-affected initial states, or different radical pairs), and make `D` more realistic using Fisher-information-style measures that include a noise model. Optimal control remains central.

### The old model, in words (this must be reproduced exactly by the new code)

- State space: 2 electron spins plus one spin-1 nucleus on radical 1 only, so dimension 12.
- Hamiltonian: hyperfine coupling on radical 1 with an axially symmetric tensor `(A(1−δ), A(1−δ), A(1+2δ))`, exchange `J (S1·S2 − 1/4)`, and Zeeman `ω0 (S1z + S2z)`. The field is along z, which is also the unique axis of the tensor, so the old results are a **single orientation**, not orientation-averaged.
- Parameters: `A_iso = 28 MHz`, `J = 1 MHz`, `ω0 = 28 MHz`, recombination rate `k = 1 µs⁻¹`, drive amplitude cap `0.25 × A_iso`.
- Drive: acts along x on both electrons, `Σ amp_i · sin(freq_i · t + phase_i)`, normalised so its peak equals the amplitude cap (L-infinity normalisation). The alternative, L2 normalisation, fixes total power so each of N components has amplitude `cap / √N`.
- Yield: average over the three nuclear states, singlet population weighted by `k·exp(−k t)`, integrated over time. DISCO uses 0 to 6 µs on 2000 points. This is exactly a Haberkorn model with equal singlet and triplet rates, so the new spin-selective recombination code must reduce to it when `k_S = k_T`.
- Initial state: pure singlet for the electrons, with each nuclear state equally weighted.

### Vocabulary

- **dCRAB-inspired optimiser**: greedy loop. Perturb all drive frequencies randomly, optimise amplitudes and phases at those frequencies, and keep the result only if it improves. This is not canonical dCRAB (no basis accumulation, no envelope, seeds come from physics), so never call it that.
- **Super-iteration**: one round of that loop.
- **Action spectrum seeding**: starting frequencies chosen from transitions between Hamiltonian eigenstates, weighted by drive coupling and singlet character (after Hiscock et al. 2017).
- **Null test**: compare an optimised result to the distribution of results from random drives.

---

## Planned structure

```
README.md       contents, experiment index, figure index, "where is X?"
CLAUDE.md       this file
model.py        hypotheses (the physics definitions)
engine.py       propagation, yields, metrics
control.py      drives, seeding, optimiser
results.py      run IDs, save/load, checkpoints, progress printing
experiments/    one script per experiment (settings, ####, working code)
figures.py      # %% cells, one per report figure, style defaults at the top
tests/          regression numbers and a slow readable reference engine
```

Key design decisions:

- **A hypothesis is data.** A `Hypothesis` holds the radicals, nuclei and hyperfine tensors, couplings, initial state (as a density matrix), reaction scheme and rates. Variants are one-line changes, for example `hypothesis.variant(initial_state=...)`. It should be quick and clear to model a new hypothesis. That is the point of `model.py`.
- **Units**: settings are in MHz and µs. Convert to angular units (`2π × MHz`) once, at the boundary, never scattered through the code.
- **Spin-selective recombination (Haberkorn)** is evolved with an effective non-Hermitian Hamiltonian `H − iK/2`, where `K = k_S P_S + k_T P_T`. That is exact for Haberkorn kinetics. Spin relaxation would need a Liouvillian and is not built yet.
- **Engine speed**: the three nuclear initial states share one propagator, so propagate them together, and both hypotheses share one drive. The drive operator `S1x + S2x` has eigenvalues 1, 0, 0, −1, so its exponential has a closed form in cosines and sines. Explain this to Laura before implementing it (rule 2), and keep the QuTiP reference engine in `tests/`.
- **Engine speed, decided at the start of milestone 3 (2026-10-06)**: `engine.py` uses the closed form in a plain loop (Strang splitting plus Richardson extrapolation), chosen for clarity. Measured at the end of milestone 3 (scratch timing script, not a recorded run): at the step used in the tests it is somewhat slower per yield than the old engine at QuTiP's default tolerances, and at twice that step slightly faster, while being far more accurate in both cases. Faster versions exist and are deliberately deferred. Add each one only as its own approved step, keeping the plain version in `tests/` as the reference: (a) stepping in the basis where `S1x + S2x` is diagonal, so the drive exponential becomes a set of phases; (b) building every output interval's propagator at once in batched NumPy, whose memory grows with the square of the dimension, so it suits larger spin systems poorly. A smaller one, (c) computing X² once per run instead of on every step, was approved and done on 2026-10-07 (`drive_exponential` now takes X² as an argument). All give only modest gains. The larger time savings are expected elsewhere: evaluating both hypotheses in one call (they share the drive), needing fewer optimiser evaluations, and running seeds in separate processes. The first was approved and done on 2026-10-08 (`yields_together`, block-diagonal): a discrimination evaluation went from about 0.69 s to 0.43 s (scratch timing, 0–6 µs, 2000 points, max_step_us 1.5 ns). Note on the step: `yields` rounds the number of substeps up, so on the 0–6 µs, 2000-point grid (output step 3.0015 ns) max_step_us = 1.5 ns gives 3 substeps, an actual step of 1.0 ns. Accuracy figures quoted for "1.5 ns" are therefore for 1.0 ns steps. A max_step_us just above 1.5008 ns gives 2 substeps. Decided 2026-10-08 (milestone 6, step 4): `optimise_drive.py` uses max_step_us = 0.00151, true 1.5 ns steps. Scratch measurement against `haberkorn_yields` over 11 drives (none, single tones 11–100 MHz, old run-3 drive, a 4-tone drive): largest yield error 1e-5 at 1.5 ns, 4e-6 at 1.0 ns, 1e-4 at 3.0 ns (rejected); 0.31 s per discrimination evaluation against 0.45 s. The engine is equally accurate up to 100 MHz. `disruption_sweep.py` (0–10 µs, 3000 points) stays at 1.11 ns steps: its next option, 1.67 ns, is not measured. Running seeds in separate processes was done in milestone 6: six processes at once each run about 15% slower than one alone (0.36 s against 0.32 s per discrimination evaluation, scratch measurement 2026-10-08).
- **Optimiser speed, deferred (2026-10-08)**: the full replay of old DISCO run 3 took 5.3 hours because most super-iterations ran Nelder-Mead to its 1000-iteration cap (about 1,500 evaluations each, against 453 in the first). Letting Nelder-Mead stop sooner (a lower iteration cap or looser tolerances) is likely the largest saving, but it changes results relative to the old runs, so it needs its own test of how much. Laura asked to keep it noted for later; do not build it until she says so. Measured 2026-10-08 (milestone 6, step 4; scratch, round 3 of old run 3 re-run from the round-1 drive, recording every evaluation): with the √B cost the round converged by itself at 513 iterations (a cap of 300 would have lost 6e-6); with the gap cost (1.5 ns steps) it hit the 1000 cap at 1449 evaluations, and a cap of 300 or 500 would have lost 2.4e-3 in D (about 6%), because the round sat on a plateau and then improved after about 800 evaluations. Neither round was kept. A lower cap therefore loses real quality in rounds that reach it. Decided 2026-10-08: the cap stays at 1000. New evidence from all 89 rounds of the exploratory averaged run (2026-10-15) points the other way: `IDEAS.md` idea 18, for Laura to decide.
- **Runs** (built in milestone 6, `results.py`): each run gets an ID `<run type>_<date>_<nn>`, such as `discrimination_2026-10-14_01`. It saves its data, its settings (with a format version) and plain quick-look PNGs named with the ID; `load_run` reads every older version, renaming and filling in settings old versions lacked (never results). Long runs save a checkpoint after every super-iteration of every seed in `results/checkpoints/<run ID>_unfinished/`; starting the script again with exactly the same settings resumes it, giving exactly the uninterrupted result. When the run is saved the folder is renamed `_finished`, and only the newest three finished runs keep their checkpoints (Laura's choice); unfinished ones are never deleted. Scripts print progress with elapsed time and estimated time remaining.
- **Figures**: raw quick-look plots stay plain and are never used in reports. Report figures come only from `figures.py`, where each `# %%` cell has a figure ID, names the run ID(s) it loads, and uses a shared default style with per-figure overrides (line widths, which elements are shown, and so on).

### Deliberately not built yet

Add these only when Laura decides the milestone has arrived, and only through the explain-first rule: orientation averaging beyond milestone 8 (built: every direction equally likely; not built: partly aligned samples), spin relaxation, QFI and quantum bounds, gradient-based optimisers, GPU backends, parallelism beyond running seeds in separate processes.

### Build order

1. Skeleton, units, tests.
2. `model.py` with Haberkorn recombination, reproducing the old yield when `k_S = k_T`.
3. `engine.py` propagation and yields, passing the regression tests.
4. `control.py` drive and optimiser, reproducing the old DISCO and APEX results.
5. Noise-aware discrimination metric in `engine.py`. Decided 2026-10-08: counting (binomial) noise, Bhattacharyya distance over three outcomes (singlet product, triplet product, not yet recombined), optimiser uses −√B.
6. Multi-seed and null tests, plus `results.py` records and checkpointing. Done 2026-10-08. Decided: a repeat changes only the random seed (default seeds 1 to 6, systematic), all starting from the action spectrum; `STARTING_FREQUENCIES_FROM = "random"` exists for testing whether seeding helps, not yet run. Null test (`experiments/null_test.py`): random unoptimised drives, 1000 by default, 0–40 MHz by default (up to 100 MHz is safe), compared by the optimised run's own measure.
7. Hypothesis library: CISS-polarised, triplet-born with asymmetric recombination, flavin-superoxide. Narrowed 2026-10-11: milestone 7 is CISS only, with placeholder parameters clearly marked. Triplet-born with unequal rates, the quantum Zeno regime, flavin-superoxide, other radical pairs and realistic literature parameters are kept in `IDEAS.md` for later. Done 2026-10-13. Decided: CISS changes only the starting state, `ciss_density_matrix(χ)` = cos(χ/2)|S⟩ + sin(χ/2)|T0⟩ (Luo & Hore 2021 eq. 8, checked against their eq. 10); CISS axis along the field; electron 1 plays the donor; recombination stays ordinary Haberkorn (their eqs. 11-12 are `IDEAS.md` idea 15). First discrimination study: singlet-born vs CISS-born (χ = 0.1π), both anisotropy 0.3, placeholder parameters (recorded runs `discrimination_2026-10-11_05`, `null_test_2026-10-13_01`).
8. Orientation ensemble and parallel evaluation. Decided 2026-10-13: the ensemble is fixed molecules with every direction equally likely (as in a frozen solution); partly aligned molecules (as in a retina) are for later. Each orientation is made by turning the molecule (hyperfine tensors and the CISS starting state) while the field stays along z and the drive along x. Speed-ups (batching, more processes) only if measurement shows the plain version is too slow. Done 2026-10-15. Built: `oriented_hypothesis` (turns tensors and the CISS start), `orientation_grid` (SciPy's Lebedev directions), batched `averaged_yields_together` (about 9 times faster; the plain loop stays in `tests/`), an optional `orientations` input on the yields, the costs and the action spectrum, `SEARCH_LEBEDEV_ORDER`/`FINAL_LEBEDEV_ORDER` in `optimise_drive.py` (50 and 194 directions; 0 = one orientation, as before), the null test averaged the same way, and a record of every Nelder-Mead call. Decided: starting-frequency spacing 2 MHz (0.5 before; one setting to change back). No processes beyond one per seed were needed. Exploratory run (recorded `discrimination_2026-10-14_01`, `null_test_2026-10-15_01`; placeholder parameters, to be repeated with real ones): in a randomly oriented sample the best drive needs 568,048 pairs against 596,140 with no drive. A scratch check found why: drives enlarge the gap in single directions but flip its sign in others, so it cancels (`IDEAS.md` ideas 19 and 20). Possible time savings found in the same run (patience 4, about 500 Nelder-Mead calls per round) are `IDEAS.md` idea 18, not adopted.
9. Quantum ceiling (QFI or a quantum Bhattacharyya/fidelity bound). Moved out of milestone 8 on 2026-10-13: it needs physics decisions first (which quantity, for two hypotheses or for estimating χ; which spin state).

The milestones are a guide, not a fixed order. If it makes more sense to do part of a later goal now, propose it to Laura and do it if she agrees. Example: in milestone 5 the metric was built for three outcomes straight away, although the third only matters once k_S ≠ k_T (milestone 7), because adding it later would mean changing the same code twice. This does not cover things in "Deliberately not built yet"; those still need Laura to decide their time has come.

---

## Tests and regression targets

Run the tests after every change. The old engine's own numerical error in `D` is about 1e-5, so compare against old numbers with a tolerance of about 1e-4, and compare the fast engine against the slow reference engine to about 1e-6.

**DISCO numbers are stale.** DISCO used many placeholder parameters. Its numbers (the yields below, and the parameters `A_iso`, `J`, `ω0`, `k`) may be used as sanity checks only and must not spread into the new repo beyond testing. Every place that uses one, in code or tests, gets a comment at that spot saying it comes from DISCO and may rely on stale placeholder parameters. Do not use "DISCO" in names in the new code. Citations of old-repo file paths are fine.

Known numbers from the old model, DISCO setup (`δ_A = 0`, `δ_B = 0.3`, L-infinity amplitude cap, 6 µs, 2000 points):

- Static gap `|Y_A − Y_B|`: 0.015544
- Monochromatic drive at 50 MHz: `D ≈ 0.0607`
- Monochromatic drive at about 11 MHz: `D ≈ 0.049`
- Polychromatic disruption advantage over monochromatic at the same peak amplitude (L-infinity normalisation, not fixed power): about 13 percentage points (APEX, which ran on 0 to 10 µs with 3000 points and 14 super-iterations; its RUN_META text saying 10 is stale)
- Discrimination amplification over static: about ×3.5 to ×3.9 (DISCO)

**Open check, not yet confirmed by Laura:** a benchmark run found a monochromatic drive at 63.5 MHz giving `D ≈ 0.0676`, which is higher than the 50 MHz value. If Laura confirms it with the old code, then 50 MHz was not the global optimum. Supporting evidence, not the confirmation: on 2026-10-08 the new slow reference engine (`haberkorn_yields`, tight tolerances, independent of the old code) gave D = 0.067627 at 63.5 MHz against 0.060693 at 50 MHz (scratch measurement, DISCO parameters, which may be stale placeholders). Until then, do not describe 50 MHz as the global optimum in any new documentation, and do not hard-code 50 MHz as "the best" in tests. Test the specific values only.

**Old-repo comparison tests are temporary.** Every test that compares against numbers from the old repo (currently `tests/test_reference_engine.py` and the last section of `tests/test_control.py`) exists only to check that the move from the old repo was done correctly. Once the new repo is complete, archive all of them: take them out of the test suite that runs after every change, and keep them in an archive folder (agree the location with Laura then). This does not cover the slow reference engine itself or the tests comparing the fast engine against it. Those stay permanently, as rule 2 requires.

---

## Conventions

- Windows machine. Python with NumPy, SciPy, QuTiP, Matplotlib.
- Give complete, runnable code, never pseudocode or placeholders.
- Open every file with `encoding="utf-8"` (a past cp1252 crash lost results). Use `pathlib` for paths.
- Anything using multiprocessing must sit under `if __name__ == "__main__":`. Set BLAS threads to 1 when running seeds in parallel.
- Do not call `matplotlib.use("Agg")` in scripts that run interactively.
- Always set random seeds explicitly and record them with the results.
- Prefer small targeted edits to working code over full rewrites.

### Scientific framing (docstrings, README, comments)

- "Disruption" means suppressing the singlet yield in the spin dynamics. Never frame it as disorienting birds.
- The polychromatic advantage for disruption is consistent across anisotropy values. Do not say it grows monotonically with anisotropy.
- Discrimination drives collapsing to near-monochromatic solutions is a result, not a limitation.
- Never quote a number in documentation that was not produced by a recorded run. If you have not run it, say so.
