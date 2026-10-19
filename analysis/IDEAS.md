# Ideas and future plans

Laura's reminder list: things we decided not to do yet, with why they might be worth doing and what to watch out for.
Nothing here is built unless it says so. When an idea is done or dropped, move it to the "Done or dropped" section
at the bottom with a date and one line on what happened.

Numbers marked "recorded" come from a saved run in `results/`. Numbers marked "scratch" come from a one-off
measurement that was not saved as a run; check them again before quoting them.

---

## Search and drive settings

### 1. Let the search reach above 40 MHz (raised 2026-10-11, kept at 40 MHz for now)

**What:** raise the starting-frequency window from 0–40 MHz to about 0–100 MHz
(`SEED_MAX_FREQUENCY_MHz` in `experiments/optimise_drive.py` and `experiments/disruption_sweep.py`).

**Why it might matter:**
- The 40 MHz limit has no physics reason we know of. It was copied from the old code (DISCO's action spectrum and
  APEX's single-frequency scan both stopped at 40 MHz).
- The useful frequencies are the pair's own transition frequencies, which come from the Hamiltonian. For the
  baseline pairs (DISCO parameters, possibly stale placeholders) the action spectrum has peaks up to about
  105 MHz, including strong ones at 63–64 and 75–77 MHz that the 40 MHz window cuts off.
- Single tones above 40 MHz gave larger gaps than our best searched drive: 50 MHz D = 0.0607, 63.5 MHz D = 0.0676
  (scratch, reference engine) against D = 0.0598 for the best of six searches (recorded,
  `discrimination_2026-10-08_02`, random seed 4).
- The search cannot easily get there by itself: each round nudges a frequency by at most ±5 MHz, and searches
  start inside the window.
- With realistic parameters (milestone 7) the transitions will move, so the window should be set to cover them,
  not left at a fixed number.

**Watch out for:**
- Engine accuracy is measured only up to 100 MHz (largest yield error 1e-5 at true 1.5 ns steps, scratch,
  2026-10-08). Above 100 MHz there are fewer simulation steps per wave, so check accuracy first.
- Widening changes which starting frequencies are picked, not just adds more (see idea 2).
- Runtime per evaluation does not depend on frequency, so widening costs no time per calculation.
- Results from a 0–100 MHz run are not directly comparable to the 0–40 MHz runs; label them.

### 2. Spacing and number of starting frequencies (raised 2026-10-11; spacing set to 2 MHz on 2026-10-13)

**What:** raise `SEED_MIN_SEPARATION_MHz` from 0.5 to about 2 MHz, and possibly use 6 frequencies instead of 4
(`NUMBER_OF_FREQUENCIES`).

**Why:** the action spectrum is binned in 1 MHz steps and its peaks span neighbouring bins. With a 0.5 MHz minimum
spacing, two of the four slots go to near-duplicates (27.5 next to 28.5 MHz). The starting frequencies the code
would pick (baseline pairs, element-wise maximum of both pairs' scores):

| Window | Number | Minimum spacing | Starting frequencies (MHz) |
|---|---|---|---|
| 0–40 MHz | 4 | 0.5 MHz (today) | 28.5, 27.5, 5.5, 14.5 |
| 0–100 MHz | 4 | 0.5 MHz | 28.5, 27.5, 75.5, 76.5 (loses the 5–15 MHz region) |
| 0–100 MHz | 4 | 2 MHz | 28.5, 75.5, 5.5, 64.5 |
| 0–100 MHz | 6 | 2 MHz | 28.5, 75.5, 5.5, 64.5, 14.5, 43.5 |

Our best drives so far lean on one strong component near 10–14 MHz, so losing that region would matter.

**Watch out for:** 6 frequencies means Nelder-Mead tunes 12 numbers instead of 8, so each round is slower
(not measured). The 2 MHz spacing is a judgement call; check the table again for realistic parameters.

**Done in part (2026-10-13):** `optimise_drive.py` uses a 2 MHz spacing from milestone 8 on (set 0.5 to compare
like with like with older runs). Six frequencies not tried.

### 3. Random-start searches: does physics-based seeding help? (built, not yet run)

**What:** one six-seed run with `STARTING_FREQUENCIES_FROM = "random"` (and `RANDOM_START_MAX_FREQUENCY_MHz`
matching the window), compared with the action-spectrum run with the same other settings.

**Why:** answers whether starting from the action spectrum actually finds better drives, or only finds them
sooner. That is the "expensive null test" (optimisations from random starts, rather than random drives).

**Watch out for:** about 2–3 hours for six seeds. Compare like with like (same window, spacing, number of
frequencies).

### 4. Single-tone benchmark scan, 0–100 MHz (raised 2026-10-11)

**What:** score single-frequency drives at the cap from 0 to 100 MHz for discrimination, by pairs needed, and save
it as a recorded run. Needs a small new script or a section in an existing one (decide where first).

**Why:** a cheap benchmark (about 1–2 minutes) that every optimised drive should be compared with. It would also
make the 63.5 MHz value a recorded result in the new code (CLAUDE.md still asks for the check with the old code).

### 5. Widen the null test to 0–100 MHz (raised 2026-10-11)

**What:** `RANDOM_MAX_FREQUENCY_MHz = 100.0` in `experiments/null_test.py`.

**Why:** a tougher luck test. The current null test only drew random drives from 0–40 MHz
(recorded, `null_test_2026-10-11_01`: 0 of 1000 beat the optimised drive). About a minute.

---

## Physics: parameters and hypotheses (milestone 7 was narrowed to CISS only, 2026-10-11)

### 11. Realistic parameters from literature sources (to do)

**What:** replace the placeholder values (from DISCO, the old repo: hyperfine 28 MHz, exchange 1 MHz, Zeeman
28 MHz, recombination 1 µs⁻¹) with literature values for a real radical pair, each cited next to the number.

**Why:** every result so far uses placeholders. The 28 MHz Zeeman frequency corresponds to a field about 20 times
Earth's, so results are not yet about a realistic compass. Laura chose (2026-10-11) to keep placeholders for now,
clearly marked, and to choose the sources later. Do not type physical constants from memory.

### 12. Triplet-born pair with unequal recombination rates (deferred 2026-10-11)

**What:** a preset starting in the triplet (equal mix of T+, T0, T−) with k_S ≠ k_T. The model already supports
both (`triplet_density_matrix()`, separate rates); no new code, just a preset.

**Why:** a natural competing hypothesis, and the first case where the three-outcome noise measure (milestone 5)
differs from the plain gap. **Rates:** the old repo has no separate rates: every script uses one rate,
`k_rate = 1.0` µs⁻¹, for both channels. Values for k_S and k_T need a source (idea 11).

### 13. Quantum Zeno regime (deferred 2026-10-11)

**What:** pairs where one recombination rate is much faster than the spin dynamics. Fast recombination acts like
repeated measurement of singlet or triplet character, which slows the singlet-triplet mixing (the quantum Zeno
effect). Builds on idea 12 (needs k_S ≠ k_T); rates need a source.

### 14. Flavin–superoxide pair (deferred 2026-10-11)

**What:** one radical a flavin with several nuclei, the other superoxide with no magnetic nuclei.

**Watch out for:**
- Real superoxide loses its spin orientation within nanoseconds (fast spin relaxation). The model cannot include
  that yet (spin relaxation needs a Liouvillian and is on CLAUDE.md's "not built yet" list). Without it, results
  describe an idealised limit only.
- A realistic flavin needs several nuclei, so the system is much bigger. Rough estimate (not measured): the cost
  per evaluation grows like the cube of the number of nuclear states, so two nitrogens instead of one would be about
  27 times slower. Measure before any long run.

### 15. CISS in recombination too, as in Luo & Hore (deferred 2026-10-11; mechanism built 2026-10-15)

**Status 2026-10-15:** `Hypothesis` now has a `recombination_state` (default the singlet), and
`flavin_tryptophan_hypothesis` uses P^R as below (Daniel's spec), with electron 1 as D. The baseline CISS runs and
presets still recombine through the singlet; switching them is a one-line `variant` if wanted.

**What:** milestone 7 changes only the starting state (`ciss_density_matrix`, the paper's eq. 8). The paper also
tilts the state the pair recombines through: P^R = |ψ_R⟩⟨ψ_R| with |ψ_R⟩ = cos(χ/2)|S⟩ − sin(χ/2)|T0⟩ (eq. 11),
differing from the start only in the sign of the spin-polarisation term (eq. 12). In our Haberkorn model, P^R would
replace the singlet projector P_S in K = k_S P_S + k_T P_T.

**Why:** our CISS results are a simplification of the paper until this is in; say so when citing it.

**Watch out for:** it needs one new piece of information in every `Hypothesis` (the recombination projector, or χ).
That changes each hypothesis's saved description, so `null_test.py` and automatic resume will no longer match runs
saved before the change. Manageable, but it is a saved-format change. Also decide which electron is the donor D:
today electron 1 plays D (so χ = π/2 means electron 1 up), and the sign of the polarisation depends on that choice.

### 17. How detectable is CISS as χ changes, with an optimised drive? (raised 2026-10-13)

**What:** repeat the six-seed run and null test at other χ (for example 0.05π and 0.25π) and plot optimised pairs
needed against χ.

**Why:** without a drive, pairs needed grow like 1/χ² for small χ (recorded, `discrimination_2026-10-11_01` to
`_04`: 54 million at 0.05π, 14 million at 0.1π). At χ = 0.1π the optimised drive needs 16,150 pairs (recorded,
`discrimination_2026-10-11_05`). Whether the optimised figure follows the same 1/χ² law, and where it becomes
impractical, is the natural next question for CISS. About 2–3 hours per χ in the background.

### 16. Other radical pairs

Different radical pairs as competing hypotheses (for example flavin–tryptophan with realistic hyperfine
couplings). Depends on idea 11.

---

## Accuracy and speed

### 6. Confirm the 63.5 MHz single tone with the old code

Open check in CLAUDE.md: a benchmark found D ≈ 0.0676 at 63.5 MHz, higher than 50 MHz. The new reference engine
agrees (scratch). Until Laura confirms it with the old code, do not call 50 MHz the best single tone.

### 7. Step size in `disruption_sweep.py`

The sweep's 0–10 µs, 3000-point grid allows 1.11 ns steps (today) or 1.67 ns (2 per output interval, faster). The
accuracy of 1.67 ns has not been measured; measure against the reference engine before switching.

### 8. Deferred engine speed-ups

Listed in CLAUDE.md under "Engine speed": (a) stepping in the basis where S1x + S2x is diagonal, (b) batched
propagators. Modest gains only; each needs its own approved step with the plain version kept as the reference.
Likely to matter more once spin systems get bigger (milestone 7 onwards).

---

### 18. Cut search time: patience 4 and about 500 Nelder-Mead calls per round (raised 2026-10-15)

**What:** `PATIENCE_SUPER_ITERATIONS = 4` instead of 6, and stop each round's Nelder-Mead after about 500 calls
(today: at most 1000 Nelder-Mead steps, which is a median of 1,168 calls per round). Possibly fewer directions while
searching too.

**Why:** worked out from the exploratory run `discrimination_2026-10-14_01` (recorded; the calculations are scratch,
from its saved histories and its record of every call):
- Patience 4, exact (a search's nudges do not depend on patience): the best of six unchanged, the worst seed 0.9%
  lower in √B, computing time 20% less, waiting time 9% less (the best seed kept improving to round 14).
  Patience 3: worst seed 9.3% lower, computing time 43% less, waiting time 13% less.
- 500 calls per round, rough (assumes later rounds start where they did): 55% fewer calls, the best of six
  unchanged, the worst seed about 4% lower. 300 calls: 72% fewer, the worst seed 13% lower.
- 50 directions while searching agreed with 194 within 2% for all six final drives; 26 or 38 not tried.

**Watch out for:**
- One aligned round (2026-10-08, gap cost) lost 6% with a cap of 300 or 500, so the averaged run may be the easy
  case. Check on an aligned run's recorded calls before adopting it for everything.
- SciPy's `maxiter` counts Nelder-Mead steps, not calls; `maxfev` limits calls. Results will differ from older runs.
- Waiting time is set by the slowest seed, so it falls less than computing time.

### 19. Drive strength for a sample (raised 2026-10-15)

**What:** score single tones at several strengths below the 7 MHz cap for the averaged sample, and possibly let the
search choose a weaker drive (cap as a ceiling, not a target).

**Why:** the peak cap scales every drive to exactly 7 MHz, so "no drive" cannot be reached. In the exploratory run,
four of six seeds ended worse than no drive and only 1.3% of random drives beat no drive (recorded,
`null_test_2026-10-15_01`).

**Watch out for:** letting the strength float changes the optimisation problem; the power cap (L2) also fixes it.

### 20. Partly aligned samples, or yields resolved by direction (raised 2026-10-15)

**What:** in a sample whose molecules are partly aligned (a retina, an oriented film), or if yields could be measured
direction by direction, drives might help far more than in a randomly oriented sample.

**Why:** scratch check (2026-10-15, 194 directions, singlet-born vs CISS-born at χ = 0.1π, placeholder parameters):
the drive optimised for aligned molecules (`discrimination_2026-10-11_05`) makes the gap 2.3 times bigger on average
in single directions, but flips its sign in 40% of directions (by weight), so 77% cancels and the sample gap falls
from 0.0035 (no drive) to 0.0019. The best sample drive flips almost nothing (2%) and changes little.

**Watch out for:** needs a model of the alignment (CLAUDE.md milestone 8 left partly aligned molecules for later),
and for direction-resolved measurement the measure changes (distances add over directions, rather than the distance
of the averaged yields).

### 21. Orientation average for a molecule with no symmetry axis (raised 2026-10-15)

**What:** `oriented_hypothesis` and the orientation average (`averaged_yields_together`) take two angles. For a
molecule symmetric about its z axis (every pair before 2026-10-15) that covers everything. For the flavin-tryptophan
pair it does not: whatever φ, the field seen by the molecule stays in its xz plane (tested, `tests/test_model.py`).
So do not average that pair with `SEARCH_LEBEDEV_ORDER`/`FINAL_LEBEDEV_ORDER`.

**Fix when needed:** with no drive, average over field directions with `field_direction_hypothesis` and the Lebedev
weights (it then covers the whole sphere). With a drive, a third angle is needed (the drive's direction around the
field), averaged too.

## Housekeeping

### 9. Mark slow tests (if the test suite keeps growing)

The suite takes about 55–80 s. If it grows much more, tag the slow tests so a quick run can skip them.

### 10. Archive the old-repo comparison tests (end of project)

CLAUDE.md: once the new repo is complete, move the tests that compare with old-repo numbers out of the everyday
suite into an archive folder (location to agree then). The slow reference engine and its tests stay.

---

## Done or dropped

- Idea 2, spacing part: done 2026-10-13 (2 MHz in `optimise_drive.py`).
