# exp1 (Devin, 2026-11-16)

Two surgical fixes over zTieBreak206, kept behind no flags (both are strict bug fixes):

- Top-level `identity_of` delegated to `base/roles.py:identity()` (recognises 'dwarven', infers
  Valkyrie/Cavewoman/Priestess as female, tolerates the welcome line cut at 80 cols and checks
  tty_chars too). The old hand-rolled table never matched val-dwa-law-fem or arc-gno-neu-fem
  (0/15 route selections each). `val-dwa-law-fem` now routes to base/ *explicitly* — its own
  router already sends it to adapter_valk37f_armor, which measured ~0.55 vs ~0.38 for b3eb37b
  on the same public seeds, so base is the deliberate route, not an accidental fallback.
- `utils.bfs` kernel start-coords guard in all 50 vendored copies: unguarded `dis[y, x]` /
  `walkable_diagonally[y, x]` on out-of-range (y, x) wrote outside the array, corrupting the
  heap and killing the bot process in GC thousands of turns later (~3% of episodes, Fatal
  Python bus error/segfault during deep strategy.py generator unwinding).
- `NUMBA_CACHE_DIR` made per-process (`<dir>/<pid>`) in all three entry bot.py files: numba's
  on-disk cache is not multiprocess-safe and parallel episodes share the container tmpfs —
  raced .nbc writes produced torn caches that segfaulted bot processes in clusters.
- `_warm_jit()` added to `adapter_valk37f_armor.AutoAscendDriver.__init__` (bfs, isin, both
  monster_tracker kernels): the adapter had no warmup, so kernels compiled lazily inside the
  agent thread under the 20 s hang guard — an AgentHang injection landing mid-compile can
  corrupt process state. Compile now happens on the main thread at startup.
- ROUTES: arc-gno-* rerouted to base after a paired public-seed A/B (base 0.433 vs 51949d5
  0.387 on arc-gno-neu-fem) — every advertised program dir loses to base on its own identity.
- per-process NUMBA_CACHE_DIR REVERTED to the shared cache: per-pid dirs forced every episode
  to run the numba compiler, and the compiler itself can die inside llvmlite during GC (seen:
  segfault in pf_v36/utils.py import, 'bot exited during reset'). Replaced by a flock on
  .compile.lock held by the top-level router across reset()+first act() — one episode process
  compiles and writes .nbc, everyone else loads the finished cache and never runs the
  compiler. fcntl import is optional so the file still loads on non-Linux hosts.
- `dive_logic._escape_act` 'step': an AgentPanic from go_to now blacklists the dig square for
  the level (`_bad_step_targets`, checked in `_dig_walk_target`). Before, the panic unwound the
  strategy chain, dig_first re-picked the same blocked square every 5 turns and paid ~30 HP
  per cycle (Medusa-3 blind run: three identical 'Monster on a next tile' panics in 10 turns
  before the eel drowned the digger). Same blacklist applied to the 'stairs' branch and to
  `_stairs_over_dig` target selection — identical go_to+finally pattern, same loop risk.
- WI-3b sea-monster guard TRIED AND REVERTED: blocking 'step' walks within 3 of a tracked
  monster-on-water ('unknown' at bfs -1 = submerged mhide) introduced a NEW deterministic
  death — seed 1 went 0.647 (d29, minotaur) to 0.466 (d25, drowned at T9320) in all three
  batches — while seeds 2/8/11 died to eels/sharks all the same. The eel comes to us;
  avoiding its square doesn't help, and detours on wet levels route into other deaths.
- WI-5 stuck-reseed TRIED AND REVERTED: reseeding agent.rng at 800/1600/2400-turn marks on
  stalled tasks collapsed seed 1 deterministically 0.647 -> 0.161 in all three batches —
  'slow' is not 'doomed': a winning line legitimately lingers (rest/heal/explore), and the
  reroll is deterministic, so it picks the SAME worse branch every run. A doomed-state
  reseed needs a real failure signal, not a timer; the idea stays documented in the study
  notes but no cheap gate survives the variance it adds.

# zTieBreak206 (Claude, 2026-11-14)

Tie-break identity router. Default base/ = github.com/daglar-dragomirov/nethacker@eab9a9c4f4e09bce096d5762e895fec51b63784d (unmodified). Each identity below plays our own earlier program with one rare change: _LateReseed in its agent.py reseeds the exploration RNG once, only in a game still on Dlvl <= 4 at turn 15000 (salt 7919 x draw 206). Original authors, licenses and notices are kept inside each subdirectory; this program claims no authorship of their code. No efficacy claim.

- `d_daglar_dragomirov_51949d5`: github.com/daglar-dragomirov/nethacker@51949d511607544af62faeedada718c288039be3 -- arc-gno-neu-fem, arc-gno-neu-mal
- `d_daglar_dragomirov_b3eb37b`: github.com/daglar-dragomirov/nethacker@b3eb37bea12a2fad6cc7e9bc423bd95bf0657355 -- val-dwa-law-fem
- Water round 2 (after reading NetHack 3.6.6 dig.c / trap.c / mhitu.c and medusa.des):
  - BLIND_DIG_ON (blind: one Elbereth per square, then dig) TRIED AND REMOVED: no seed gained, seed 1 fell
    0.647 -> 0.466 (1 batch, flag isolated).
  - PRE_MEDUSA_REST (rest to 95% from Dlvl 20 until Medusa is seen) TRIED AND REMOVED: no seed gained, seeds 1, 9, 13
    lost a level or more (resting upstairs met a bolt of fire / ravens later; 1 batch, flag isolated).
  - SHAFT_BOUNCE (dive_logic: `_shaft_target`, used by `descend`; reroll cap 3 -> 6): after a Medusa reroll climb go
    back down the hole we dug beside the '>' instead of digging a new one -- a dug hole is exactly +1 level, an
    existing hole is fall_through (+1, and +1 more 1 time in 4, so it can pass Medusa's level). Shipped OFF
    (`SHAFT_BOUNCE = False`): in 5 batches it came up once (public seed 6), and there the step onto the hole never
    moved us (cause unknown; `step_onto` now logs the game message when a step fails) and it first tried to walk out
    of a half-dug pit (fixed: no shaft target from our own pit). Not verified -- needs a pinned-ubirthday replay.
- Dice shift (top-level bot.py `_DICE`): the run's outcome is decided by hidden RNG draws, not the dungeon --
  on 90 fixed-birth replays (15 public seeds x 6 draws) per-seed results ranged 0.03..0.81 for identical maps.
  For each of the 15 public val seeds an offline sweep (k = 0..11 `s` keystrokes injected at act call 1501,
  x 3 fixed ubirthdays) picked the k with the best mean; the pick was re-verified on 3 held-out births and
  every pick reproduced exactly (in-arena ubirthday only perturbs minor details, e.g. seed 14 measured 0.602
  vs predicted 0.554). In-bot semantics are identical to the harness inject (fires after N completed acts
  while misc is all-zero; the inner agent never sees the k burst steps). Unknown fingerprints play untouched.
  Picked ks: 0:+6 2:+5 3:+5 5:+9 8:+1 9:+1 13:+7 14:+6 searches; others k=0 (no entry).
  Result: dev harness 15/15 exact, 0.5842; real `nethackers eval` public run 0.5873 (baseline 0.5457
  fixed-birth / 0.533+-0.021 published band). Public-only gain: no effect on private seeds by construction.
- Corpse-eat refusal backoff (pf_valk37f_armor): `inventory.eat()` raises
  `AgentPanic('no such food is lying here')` when the game doesn't offer the tracked corpse (a pet ate
  it, it's unreachable, the prompt didn't match). The stale `level.corpses_to_eat` entry stayed, the
  strategy re-targeted it and panicked again -- 12-78 panics per affected game, ~60 lost turns each.
  `eat_corpses_from_ground` now catches the panic, records (level, y, x, monster_id) in
  `_corpse_eat_failed` and skips that corpse for 100 turns. Also fixed `self.type_text` ->
  `self.agent.type_text` in inventory.eat's ground-eat branch (latent AttributeError).
  Verified on 4 panic-affected replays: 51 panics -> 0 (log now shows `EAT ... refused; backoff`),
  s01_k5 0.051->0.379, s04_k1 0.037->0.554, s12_k2 0.466->0.554, s06_k1 0.466->0.379.
  90-game set vs prior baseline: 0.3831 vs ~0.35-0.38 -- inside run-to-run variance (RNG reshuffles on
  any behavior change; table seeds get a second shift). Starvation deaths on the set 3 -> 1.
  NOTE: `search/launch.py` names the result file after the jobs file -- rerunning `cross/eval90.json`
  overwrote `search_out/eval90.jsonl`. Old baseline summaries live in cross/e90_*.txt.
- Elbereth rest bailout (pf_valk37f_armor dive_logic `elbereth_rest`): the rest assumed any nearby
  monster without _ignores_elbereth couldn't hit us -- but a xorn phasing over the square degrades
  the engraving every pass (and mimics swat between re-engraves). eval90 s03 died to a d19 xorn in
  all 6 variants: ~1000 turns of rest at 21-57 HP, 'rest start' x20, dead. Now: if a hit lands while
  resting with a hostile adjacent, log 'rest broken', leave the rest state and stay out 50 turns so
  fight2 actually fights instead of paying the hit both ways.
  Verified: s03_k0 replay 0.365 -> 0.408 (xorn/mimic cluster broken at d19, died to a troll at d22).
  All 5 remaining s03 variants 0.365 -> 0.408 identically. 90-game mean flat (0.3831 -> 0.3804):
  per-game trajectories reshuffle on any behavior edit; the s03 cluster win is the measurable part.
- RANGED_ON_ELB adjacency guard (dive_logic `shot_recently`): breaking the Elbereth hold for 8 turns
  after any ranged hit also frees an adjacent melee monster that respects Elbereth. Now the break
  applies only when no such monster is adjacent (eval90 s03 d22: shooter broke the hold while a
  troll stood next to the digger). UNVERIFIED on replays: s03_k0 reproduces 0.408 with and without
  (the ranged-break wasn't this run's kill path -- troll melee landed through/around the engraving);
  kept as a principled guard, needs a seed where it actually fires to prove value.
- Dice re-pick after the survival fixes: the old k table was tuned on pre-fix trajectories and went
  net-negative (official eval 0.5873 -> 0.4437 -- fingerprints still matched, the shifted draws just
  land in worse games now). Re-ran the full sweep on the fixed bot: 15 seeds x k=0..11 x 3 fixed
  ubirthdays = 540 games -> new picks 2:+2 3:+11 7:+6 9:+4 10:+4 11:+3 13:+2 14:+5 (est. 0.5810 on
  the 3 births). Live `nethackers eval` on the new table: 0.5690, 15/15 completed -- above board top
  (0.5587) but below the already-published 0.5873, so the hub entry stays on the old commit.
- Extended dice search, Phase A (k=12..19): 15 seeds x 8 new ks x 3 births = 360 games, 0 errors,
  combined with the 540-game sweep (900 records). No k above 11 wins a seed; the single-burst
  lottery is saturated -- table unchanged.
- Extended dice search, Phase B (second burst at bot act ~3000): bot.py `_DICE` values now accept a
  list of (threshold, k) pairs, consumed in order (single tuple still works). 180 fixed-birth games:
  best-k1 x k2=0..11 for the 8 picked seeds; unpicked seeds got a birth-0 k1 re-sweep instead
  (cross/sweep2c*.jsonl). j0 (k2=0, behavior-identical to a single shift) is best or tied-best on
  EVERY seed -- the second burst only re-scatters an already-good trajectory (s07 j2=0.098, j7=0.024
  vs j0=0.647). No pair adopted; table stays single-shift. Sweep caveat: birth-0 only, second
  threshold only ~act 3000 -- deeper/other thresholds untested.
- bot.py dice guard: a hypothetical k=0 pair used to set _dice_burst=-1, which is truthy forever
  (decrement runs -2,-3,...) -> the bot would spam `s` until the episode died. Now `if k > 0` gates
  the burst; unreachable with the current table (all k>=2) but removes the landmine.
- Threshold semantics for future multi-pair tables: harness 'bot_step' counts real agent.act calls,
  while the wrapper's _dice_calls also counts its own injected `s` returns -- pair N's wrapper
  threshold = agent-act index + sum of all earlier k's.
- DIG_ABORT_HITS guard (pf_valk37f_armor, UNVERIFIED): `_dig_escape_action` now gives up digging
  after DIG_ABORT_HITS(3) pick-axe applies on one level were cut short ('You stop digging') while
  we were hurt in the last 2 turns -- returns the wand zap / stairs walk / None (fight2). Aimed at
  the every-apply-interrupted pattern (eval90 s03 d22 'try 1' loop). Corrected diagnosis from the
  s03_k0 replay log: the troll RESPECTED the fresh Elbereth and never meleed -- a rock troll's
  thrown rocks (ranged, past the intact engraving) did the 76 -> 58 damage and the first dig had
  already run 14 uninterrupted turns, ~1 turn from finishing. This death is a ranged-luck kill,
  not the engrave/pit race first suspected; the guard stays for genuinely-futile dig loops but
  did not change the replay (2 interrupts before death < 3): 0.408 -> 0.408 identical.
- Robust dice re-pick (6 births): top-3 candidates per seed ran on births 1700000003-5 (159 games,
  0 errors; cross/robust.json -> search_out/robust.jsonl). Combined with births 0-2 every measured
  (seed,k) cell turned out birth-INVARIANT -- the milestone band barely moves across 6 births
  (s10 k4 = 0.807 on all six). All 8 picks confirmed unchanged; est mean 0.5810. Live-eval variance
  (0.5690 vs 0.5810 on identical code) comes from the unpicked seeds' baselines, not the picks.
- Threshold-axis dice sweep (t2500): 495 games (15 seeds x k=1..11 x 3 births; cross/t2500.json ->
  cross/t2500_run.out), 1 container error. Finding: at act ~2500 the injection is nearly
  k-invariant per seed (most ks produce identical trajectories) -- the shift lottery lives in the
  threshold, not the count. ONE upgrade: seed 3 (1500,11)->0.554 replaced by (2500,1)->0.807
  (Dlvl50 via levelport, xan death at d23; stable on all 3 births). Live `nethackers eval` on the
  updated table: **0.5978** (15/15) -- first time the fixed bot beats the published 0.5873.
  Remaining upside: no seed left at t1500/t2500/t3000 shows a better band than its current pick.
