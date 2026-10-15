# Hidden-pool refresh and final re-score

Status: current design, implemented by `tools/pool-refresh.sh` and
`tools/router/router.py`. Contest length two weeks, unlimited submissions per
team.

## Why

A candidate engine necessarily sees the scored prompt at run time. The sealed
record names a prompt only by its sha256, but it can carry engine-chosen
values. Timing is a slow side channel too, and teams may submit without limit.
A prompt learned that way could be hard-coded into an engine for a fake
speed-up. Three layers limit the damage:

1. **Every live golden per job**: each ranked job measures every prompt of
   the fixture's `live_goldens`, one pair per prompt, and scores the mean of
   the per-pair composites.
2. **Pool refresh every 3-4 days** (4 refreshes in two weeks): the live pool is
   replaced by 8 prompts no submission has seen.
3. **Final re-score**: at the close, the top submissions are re-measured on a
   held-out pool that was never staged on a box during the contest. The final
   ranking comes only from this. Anything leaked during the contest is worth
   nothing at the end, and day-to-day prompt drift does not decide the winner.

## Prompt generation (both the refresh and the held-out pool)

Prompts must not be recoverable from a leaked fragment. The first live pool's
prompts are single passages of public-domain books, so a few leaked tokens
identify the book and therefore the whole prompt. New prompts are **stitched**:

- A corpus directory of public-domain works (plain text), kept in
  `.mlxfast-private/corpus/` (never in git), large enough for ~40 prompts.
- Each prompt is built from paragraphs drawn at random from several different
  works (at least 4 works per prompt), in random order, until it reaches the
  fixture's `seed_tokens` length with the target tokenizer
  (`tools/author-prompts.py` does the final trim). A leaked fragment identifies
  one paragraph, not the prompt.
- Randomness from `/dev/urandom`. The generator records, privately, which work
  and paragraph each piece came from, so a prompt can be audited.
- Every generation gets a label (`g1`, `g2`, …, and `final` for the held-out
  pool); prompt names are `<label>-<n>` (for example `g2-5`).

## Refresh (`tools/pool-refresh.sh`, run from the organizer workstation)

Idempotent steps, each refusing on any mismatch:

1. **Author**: 8 stitched prompts for the next generation (above).
2. **Record**: serial golden plus MTP-1…6 oracles for every prompt
   (8 × 7 = 56 goldens), spread across the fleet with `tools/record-goldens.sh`,
   each one on **two** boxes and required to be byte-identical (the engine is
   deterministic; a mismatch means a broken box or engine).
3. **Stage**: install the new goldens next to the current ones on all 8 boxes
   (`/srv/bonsai9b/goldens`, root:bench 0750), verified by sha256. The old
   generation stays in place, so in-flight jobs are unaffected.
4. **Pin and switch**: one commit to the trusted fixture that pins the new
   goldens and sets `live_goldens` to the new generation, pushed to `main`
   (and to the armed base branch the dispatcher overlays onto). Jobs
   dispatched after the push use the new pool; jobs already running keep the
   pool of the commit they were dispatched at.
5. **Verify**: the ranked preflight on every box passes with the new fixture
   (pins staged and matching) before the script reports success.
6. **Retire**: after all jobs on the old generation have finished, remove its
   goldens from the boxes; keep a private archive (for disputes).

Timing: fleet time is spent in step 2 (and a short drain). The script
records its progress so a crash resumes where it stopped. It never touches the
calibration (see below).

Calibration: the box calibration file carries one band per live golden
(`tools/calibrate-box.sh`), and benchd gates each control leg with the band of
the prompt it measured. A new generation has no bands yet, so every box must be
calibrated again on the new live goldens before the pin commit is pushed;
otherwise the ranked preflight refuses every job. The pin step also refuses an
`official_pairs` that is not a multiple of the generation size, because every
job measures every live golden.

## Final re-score (`router.py rescore`, operator-run at the close)

The final pool is **not yet authored** for this track. When the contest is
ready to close, follow the steps below.

1. **Held-out pool**: author generation `final` — 8 stitched prompts × 7
   goldens = 56 goldens, each agreed byte for byte by two contest Sparks.
   Keep it off the boxes until the freeze: goldens and prompts live only in
   `.mlxfast-private/final/` on the organizer workstation (never in git, never
   staged, never pinned in a pushed fixture) until the close.
2. **Freeze**: at the deadline the router stops accepting submissions.
3. **Select**: for each team, its best submission by mean contest score over
   runs that scored or failed because of the submission (a team failure counts
   as 0.0; infrastructure runs are excluded). Ties go to the most recent
   submission; then the top N teams (N set by the organizer; default 10).
4. **Stage and pin**: stage the `final` goldens on all boxes; build a
   `rescore/base` branch = the contest's final base plus one fixture commit
   whose `live_goldens` are the `final` generation.
5. **Re-measure**: each selected submission's editable paths are overlaid onto
   `rescore/base` (the same REPLACE overlay as normal submission) and dispatched
   K times (default K = 5), one run per slot, spread across different boxes, so
   every submission is scored on several `final` prompts and boxes.
6. **Slots and retries**: each team has exactly K slots. A slot's value is its
   first run that ends with a score or a team failure (replay rejected, floor or
   correctness failure, refusal, unsealed, `watchdog_timeout`, or a watchdog
   cancellation because the engine hung). A run whose prompt is not `final-*`
   counts as infrastructure, not as a score. Infrastructure outcomes do not fill
   a slot; `router.py rescore-retry` re-dispatches those slots. Per slot, total
   infrastructure attempts = `1 + --max-infra-retries` (default 2 extra attempts
   after the first, so 3 tries). The first candidate resident boot failure in a
   slot may be re-dispatched once (separate from the infra budget); a second
   boot failure in that slot counts as 0.0. After setup, a control-leg failure
   (`control_failed`) or band refusal (`band_refused`) that still fails after
   those attempts counts as **0.0** for that slot (status `penalized`), because
   the submission already ran and the fault is induced by the engine afterward.
   Pre-setup infrastructure (preflight refusal, setup failure, manual cancellation
   not caused by the watchdog, or off-pool) leaves the slot **unresolved** when
   retries are exhausted.
7. **Rank**: the mean is the average of the K slot values (the score, 0.0 for a
   team failure, or 0.0 for a penalized post-setup infrastructure slot). A team
   with any slot still pending or unresolved has no mean and status `incomplete`;
   incomplete teams are listed after every complete team and are never ranked
   above them. `rescore-report` lists every run and, per team, each slot's run
   ids, outcomes, and value so the ranking can be audited.

## Checks

- A generated prompt never contains a paragraph that appears in any earlier
  generation or in the held-out pool (the generator refuses reuse).
- The refresh never stages the `final` generation; the rescore refuses to run
  if any `final` golden was ever staged before the freeze (the staging log is
  the evidence).
- Both scripts have `--dry-run`, and a test suite with stubbed boxes.

## Where the goldens live

| Copy | Holds | Purpose |
|---|---|---|
| Each Spark, `/srv/bonsai9b/goldens` (root:bench, unreadable by the sandbox user) | the live pool (and the next generation once staged) | the only copy scoring reads; a ranked job never fetches goldens |
| Jump box, `<jump-box-goldens>/live` (0700) | the live pool, with `SHA256SUMS` from the fixture pins | the source for restaging a rebuilt box |
| Organizer archive (a private repository, organizer only) | the live pool, every retired generation, the held-out `final` pool, the paragraph ledger | backup and dispute evidence |

The held-out `final` pool never goes to the jump box or a Spark before the
freeze. Goldens are never committed to the challenge repository, because the
sandboxed participant code can read the ranked checkout. After each refresh,
copy the new generation to the jump box and the archive. Before restaging a box
from the jump box, verify with `sha256sum -c SHA256SUMS`.

## Runbook (organizer commands, from the organizer workstation's checkout)

Tools: `tools/stitch-prompts.py` (authoring), `tools/pool-refresh.sh` (refresh),
`tools/router/router.py freeze|rescore|rescore-retry|rescore-report`. Private state lives
under `.mlxfast-private/`: `refresh/<label>/` per generation (state file,
prompts, provenance, per-box recordings, agreed goldens, `pins.json`),
`final/` for the held-out pool, `refresh/ledger.json` (every paragraph ever
used) and `refresh/staging.log` (every golden installed on a box, with time).

**Required env (no default):** `REFRESH_BOXES` (space-separated ssh hosts, at
least two) and `REFRESH_BOX_REPO` (checkout path on each box, relative to the
box home; for example `bonsai9b/repo`; must be at a commit that has
`record-goldens.sh --only`).

**Defaults in `tools/pool-refresh.sh`:** `REFRESH_BOX_GOLDENS` →
`/srv/bonsai9b/goldens`; `REFRESH_GPU_LOCK` → `/tmp/mtplx-gpu-exclusive.lock`;
`REFRESH_ARMED_BASE` → `loadtest/base` (overridable; see below). **`REFRESH_BOX_SNAPSHOT`**
is optional: when unset, recording uses the box login environment's
`MLXFAST_TARGET_SNAPSHOT_DIR`. Stitching needs the `tokenizers` package:
`REFRESH_PYTHON` may name a python that has it (the pinned image does).

**Before the contest: author and record the held-out pool** (future work for
this track; repeat for a replacement pool or the next contest).

```bash
export REFRESH_BOXES='<box-a> <box-b>'
export REFRESH_BOX_REPO='bonsai9b/repo'
tools/pool-refresh.sh --label final --dry-run --corpus .mlxfast-private/corpus --snapshot <SNAPSHOT>
tools/pool-refresh.sh --label final --corpus .mlxfast-private/corpus --snapshot <SNAPSHOT>
# -> .mlxfast-private/final/{prompts,goldens,pins.json}; nothing is staged or pinned
```

**Each refresh (every 3-4 days), generation g<N>:**

```bash
tools/pool-refresh.sh --label g2 --dry-run --corpus .mlxfast-private/corpus --snapshot <SNAPSHOT>
tools/pool-refresh.sh --label g2 --corpus .mlxfast-private/corpus --snapshot <SNAPSHOT>
#   author -> record (56 goldens x 2 boxes, byte-identical) -> stage (all boxes) -> pin
#   a crash or a failed box: fix it and re-run the same command; finished work is skipped
#   one step alone: --step author|record|stage|pin|verify|retire
# review the fixture diff, then run the printed commands:
git add fixtures/bonsai2_9b_cuda_v1_track.json
git commit -m "pool refresh: pin g2 and switch the live goldens"
git push origin HEAD:main && git push origin HEAD:<armed-base>
tools/pool-refresh.sh --label g2 --step verify      # every box matches origin/main's pins
# once no job dispatched before the switch is still queued or running:
tools/pool-refresh.sh --label g1 --step retire      # unpins g1: commit and push as printed
tools/pool-refresh.sh --label g1 --step retire      # after that push drains: removes g1 from the boxes
```

The pin commit sets `live_goldens` to the new prompts and
`hidden_correctness_golden` to the serial tape of `<label>-1`. The old
generation stays pinned and staged until it is retired, so
jobs queued at the old commit keep passing. The ranked preflight tolerates
unpinned `g<N>-*` goldens (staged ahead of their pin) and nothing else, so a
`final-*` golden on a box before the close fails every job.

The armed base branch is the branch the dispatcher overlays submissions onto.
Its name is configurable: `router.py --base` for the router and
`REFRESH_ARMED_BASE` for the refresh script (both carry a default in the code;
set them to the branch you arm). `<armed-base>` above stands for that name.

A byte mismatch between the two boxes that recorded a golden stops the
record step before anything is pinned. Find the bad box (`rec-<job>.log`,
determinism probe), delete that job's `done job <key>` line from
`.mlxfast-private/refresh/<label>/state` and re-run.

**At the close:**

```bash
tools/router/router.py freeze
tools/router/router.py status        # wait until nothing is queued, dispatched or running
tools/pool-refresh.sh --label final --step stage --final-after-freeze <router-state>/FROZEN
tools/router/router.py rescore --base-ref origin/main --top 10 --runs 5
git -C <router-checkout> push origin rescore/base   # as printed; the sub/ branches carry it anyway
# repeat until rescore-report is complete (no incomplete teams):
tools/router/router.py rescore-retry --tag final --max-infra-retries 2   # infra + one boot retry per slot
tools/router/router.py rescore-report --wait    # <router-state>/rescore-final/rescore.{json,html}
```
