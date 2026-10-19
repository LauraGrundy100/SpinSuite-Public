# Handover: Daniel's FAD/Trp system, for the Friday noon meeting

Written Thu 15 Oct 2026, about 16:45, for a Claude Code session in the SpinSuite repo.
Meeting with Daniel and Bruin: Fri 16 Oct, 12:00. Laura wants something to show.

## Read this first

What the author of this note could and could not see:
- Seen: README.md, CLAUDE.md, IDEAS.md, and excerpts of optimise_drive.py and null_test.py (via the project sync).
- Not seen: the source of model.py, engine.py, control.py and results.py. Nothing of Laura's code was run.
- Only thing run: a scratch check of Daniel's tensors (numbers below). They are not recorded runs.
- So every statement here about how model.py is structured is a guess. Read the source first and correct this note where it is wrong.

How to work (CLAUDE.md applies in full, these are the rules that bite tonight):
- Rule 2 and 7: before building anything, send Laura ONE short plain-language plan (the pieces below, what each does, how she can check it) and wait for her yes. A technical description she did not follow does not count as a go-ahead. If she says she is lost, stop building.
- Rule 3: do not create files without asking. Likely new files tonight: one experiment script for the field scan, and figures.py. Ask for both in the plan.
- Rule 1: plain functions and dataclasses, nothing clever. Run the tests after every change. Update the README in the same step.
- Rule 6: the Hypothesis description changes, so bump the saved-format version. Old runs must still load.
- Do not type physical constants from memory (IDEAS idea 11). Use scipy.constants and print the value used.
- Never quote a number in documentation that did not come from a recorded run. Say what you have not run, in the first paragraph of every report.
- Windows, utf-8 everywhere, seeds recorded, multiprocessing under if __name__ == "__main__".
- If a decision of Laura's looks wrong, use the five-part structure from CLAUDE.md and wait.

## Goal

Show Daniel results on the realistic system he sent, in this order of value:
1. Static (no drive) yield against field direction for classical and CISS pairs, with the min and max directions and the contrast (max minus min). Compare the min and max directions with FN5 eigenvectors, as Daniel suggested.
2. Contrast metric in the code (his main ask on 9 Oct).
3. One reduced optimisation run with the contrast metric. Not converged is fine. Laura wants to start to see ideas.

If time runs short, 1 alone is a good result. Do not start 3 until 1 and 2 pass their tests.

## Facts checked (scratch script, recompute them in tests)

Daniel's tensors are symmetric, in rad/us. Divide by 2 pi to get MHz.
- FN5: trace 276.43 rad/us, isotropic part 14.665 MHz, eigenvalues -2.81, -2.43, 49.24 MHz. Nearly axial. Unique axis (eigenvector of 49.24 MHz): (-0.543, -0.778, -0.315) up to sign.
- WN1: trace 169.86 rad/us, isotropic part 9.011 MHz, eigenvalues -1.78, -1.49, 30.30 MHz.
- EED: traceless, axial about z. xx = yy = 9.55 MHz, zz = -19.11 MHz.
- Electron Larmor frequency at 50 microtesla is about 1.40 MHz (scratch). The old placeholder 28 MHz is about 20 times that.
- Survival exp(-k t) with k = 1 per us is 0.25 percent at 6 us and 6e-6 at 12 us.

## The new system

Dimension 2 x 2 x 3 x 3 = 36. Two electrons, two spin-1 nitrogens.
- Nucleus 1 couples to electron 1 through FN5. Nucleus 2 couples to electron 2 through WN1.
- Electron-electron dipolar term S1 . EED . S2 (spin operators, hbar = 1). Exchange J = 0 (assumption, see below).
- Classical pair: chi = 0, kr = kf = 1 per us. CISS pair: chi = pi/2 (maximal), kr = 10, kf = 1 per us.
- Frame: all three tensors are in one molecular frame. The field is rotated through it.

## Pieces, in order

P1. Model
- Allow a nucleus on either electron, full (non-axial) 3x3 hyperfine tensors, and the EED term. Check how Nucleus and Hypothesis store things and what axial_hyperfine_tensor_MHz does today.
- Store tensors in MHz (Daniel's values divided by 2 pi once, at the boundary, with a comment saying they come from Daniel, 15 Oct 2026, in rad/us). Never multiply twice.
- Zeeman frequency from field strength and the electron gyromagnetic ratio in scipy.constants.
- oriented_hypothesis and rotation_matrix: check that they turn FN5, WN1, EED and the CISS axis together and do not assume one axial tensor. The old two-angle (theta, phi) turning is exact for one axial molecule only. The new molecule has no single axis, so say plainly which third-angle convention is used. For the static scan the third angle does not matter.
- The drive operator S1x + S2x acts on electrons only, so the closed-form drive exponential still holds. Check engine.py does not assume 12 dimensions.
- Tests: tensor checks above, 36-dimensional fast engine against the slow reference engine to about 1e-6, old regression tests still pass.

P2. Recombination projector and rates (IDEAS idea 15, now with a spec from Daniel)
- Start ket: cos(chi/2)|S> + sin(chi/2)|T0>. Recombination projector Pr: onto cos(chi/2)|S> - sin(chi/2)|T0> (Daniel's recombination_projector_ciss, which is the start state with -chi). The nuclei are fully mixed, identity on them.
- Reading of the rates (confirm with Daniel): recombination at rate kr through Pr, spin-independent forward or separation at rate kf. Then K = kr Pr + kf 1 = (kr + kf) Pr + kf (1 - Pr). The existing propagation then works with P_S replaced by Pr, k_S = kr + kf, k_T = kf. This matches the 9 Oct notes (singlet channel recombination plus separation, triplet channel separation only).
- Yields must change. Recombination yield = kr times the integral of Tr(Pr rho) dt. Forward yield = kf times the integral of Tr(rho) dt. The rest has not reacted yet. The existing yields() returns k_S times the integral of Tr(P_S rho), which would be (kr + kf) times the right thing here. Do not silently reuse it.
- Tests: chi = 0 reproduces the old singlet path. The three outcomes sum to 1. Extend the slow reference (haberkorn_yields) to a general projector and agree to about 1e-6.
- Daniel asked if the code is wavefunction or density-matrix based. Answer for Laura: wavefunction per ket, initial_kets splits rho0 into weighted kets (here nine nuclear basis states, weight 1/9 each, with the CISS electron ket) and uses the non-Hermitian H - iK/2.

P3. Static field-direction scan (no drive)
- For chi = 0 (kr = kf = 1) and chi = pi/2 (kr = 10, kf = 1): recombination yield over a grid of field directions in the molecular frame.
- Report per pair: min and max yield, their directions, contrast (max minus min), and the angle from each of those directions to the nearest FN5 eigenvector.
- Integration window 12 us (Daniel: 10 to 12 lifetimes, lifetime about 1/kf = 1 us). The old runs used 6 us. Print the unreacted remainder at the end and check it is much smaller than the yield differences.
- Optional independent check in tests: the static yield from a direct linear solve instead of time stepping. Explain first if you propose it.

P4. Contrast metric
- Contrast = max minus min of recombination yield over a short list of directions. Primary metric for now (Daniel, 9 Oct). Bhattacharyya stays as it is, nothing to change there.
- Cost for discrimination: minus |C_A - C_B|, where A and B are the two hypotheses under the same drive. This covers "abolish the effect in one, leave the other alone".
- Directions for runs: the min and max directions from P3, or two FN5 eigenvectors (Daniel's shortcut). If the pair found for A and B differ a lot, say so and propose the union and its extra cost.
- Always report contrast with no drive next to contrast with the drive. That is the baseline.

P5. Reduced optimisation run (use optimise_drive.py, change the settings block only if possible)
- Hypotheses: A = classical (chi 0, kr = kf = 1), B = CISS (chi pi/2, kr = 10, kf = 1).
- First time ONE cost evaluation at the new size (36 dims, nine kets, 2 directions, 2 hypotheses, 12 us window). Reference points from the README, for the old 12-dimensional system: about 0.3 s for one pair in one direction, 2.1 s for 50 directions, about 500 Nelder-Mead calls per round, longest seed of the last averaged run 16.5 h. IDEAS suggests two nitrogens are about 27 times slower. That is an estimate, not a measurement.
- Time per round is roughly (Nelder-Mead calls) x (seconds per evaluation). For example, if one evaluation took 10 s, 500 calls is about 83 minutes per round. That is an illustration, not a measurement.
- Budget: launch by about 20:00, whatever has finished by about 09:00 is what Laura shows. Checkpoints save after every super-iteration, so partial results are usable. Make sure resume works.
- Knobs, in this order: fewer super-iterations (3 to 5), three to five seeds in separate processes with one core left free (Ryzen 5 3600, 6 cores), 2 or 3 frequencies instead of 4, then patience. Do not loosen the Nelder-Mead tolerances without asking Laura, it changes behaviour.
- Present the chosen settings and the expected wall time in plain language and wait for her yes before launching.
- Starting frequencies: keep the action spectrum (check action_spectrum copes with 36 dimensions and a non-axial molecule) and use the recorded random-start option only if she asks.
- Cheap extra if there is time: a single-tone scan of contrast against drive frequency for A and B at the cap (IDEAS idea 4 style). It gives a benchmark for the optimised drive and is deterministic.

Figures (ask first, rule 3): one figures.py, one cell per figure, rebuilt from saved results by run ID. Wanted: (1) yield against field direction for A and B with min and max marked, (2) contrast with and without drive, optimised against best single tone if done, (3) convergence of the reduced run. Quick-look plots from results.py are fine meanwhile.

## Assumptions to mark in code comments and tell Laura (each is an open question for Daniel)

- Rates: K = kr Pr + kf 1, and the observable is the recombination yield.
- J = 0. Daniel gave only EED, and on 9 Oct said to neglect exchange for separated radicals.
- Field: 50 microtesla geomagnetic, direction rotated in the molecular frame. Daniel has not given a strength.
- Drive: along lab x, perpendicular to the field, with the molecule turned (the existing convention). For a non-axial molecule the azimuth of the drive about the field matters, and it is not settled.
- Peak cap: the old 7 MHz was 0.25 x the old 28 MHz hyperfine. Both obvious replacements are far above realistic RF amplitudes. Suggested default tonight: 0.25 x the Zeeman frequency (about 0.35 MHz), computed from the field and not hard-coded. Laura decides. Label results as placeholder-cap.
- CISS axis: the unique EED axis, which is the molecular z in the frame the tensors are given in. Electron 1 is the donor, as in the existing ciss_density_matrix. Daniel's snippet puts T0 along the spin quantisation axis, so confirm which frame he means.
- chi: pi/2 is maximal. No value is given for a physically motivated chi.
- Integration window 12 us.

Typos in Daniel's snippet: rh0 should be rho0, and Pr is built with dims [2,2,2,2] where the system is [2,2,3,3].

## Not tonight

Robustness (hyperfine uncertainty and the Bhattacharyya priors), many frequencies and initial conditions, workstation or cluster runs, spin relaxation, partly aligned samples, orientation averaging over drive azimuth, other hypothesis pairs (CISS against Zeno), GPU, gradient optimisers. All of these are in CLAUDE.md "Deliberately not built yet" or IDEAS.md.

## Report back to Laura at the end

Short, no tables. What was built and where, what was run and what was not, the numbers with their run IDs, the assumptions list, and what she should ask Daniel. Update the README status, contents table and run list in the same step. The contents table still says milestone 8 is in progress, fix that too.

## Appendix: Daniel's message (verbatim)

Tensors, in rad/us, usable without further scaling when time is in us:

FN5 = np.array([[ 79.48092385, 137.59553919,  56.68450732],
          [137.59553919, 181.20322632,  79.23527745],
          [ 56.68450732,  79.23527745,  15.748773  ]])

WN1 = np.array([[ 51.01430479,  16.46786544,  90.57082545],
          [ 16.46786544,  -6.75181867,  23.32643811],
          [ 90.57082545,  23.32643811, 125.59269667]])

EED = np.array([[ 6.00327982e+01,  0,  0],
        [ 0,  6.00327982e+01, 0],
        [ 0, 0, -1.20065596e+02]])

Classical radical pair: kr = kf = 1/us. Rotating the geomagnetic field in the molecular frame gives the directions of minimal and maximal recombination yield. Further exploration can be limited to those (or two eigenvectors of FN5, which should be close to the minimal and maximal directions).

CISS: change the initial state and the projection operator for recombination and yield. chi = 0 is no CISS, chi = pi/2 is maximal CISS. For CISS one usually assumes kr = 10/us and kf = 1.

up = qt.basis(2,0)
down = qt.basis(2,1)
singlet = (qt.tensor(up,down) - qt.tensor(down,up))/np.sqrt(2)
triplet0 = (qt.tensor(up,down) + qt.tensor(down,up))/np.sqrt(2)

def initial_state_ciss_ee(chi):
    return np.cos(chi/2) * singlet + np.sin(chi/2) * triplet0

def ciss_projector_ee(chi):
    psiI = initial_state_ciss_ee(chi)
    return psiI * psiI.dag()

def initial_state_ciss(dims, chi):
    return qt.tensor([ciss_projector_ee(chi)] + [qt.identity(d) for d in dims[2:]])

def recombination_projector_ciss(dims, chi):
    return initial_state_ciss(dims, -chi)

rho0 = initial_state_ciss([2,2,3,3], chi)
rho0 = rh0 / rho0.tr()
Pr = recombination_projector_ciss([2,2,2,2], chi)

Daniel adds that he does not remember whether the simulation is wavefunction or density-matrix based (if wavefunction, the initial state is initial_state_ciss_ee), and that it would likely be best to meet and tweak the code together.
