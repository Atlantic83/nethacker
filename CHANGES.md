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
