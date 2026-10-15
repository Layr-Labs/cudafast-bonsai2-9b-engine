# benchd decode metric: `noop_decode_speedup`

## Context

benchd commit `e7cd2e9` expects a `noop_decode_speedup` field on each
`timed_prompt_pool` entry when a job uses **measure-job**. That field tells
benchd how to treat decode speedup on prompts that are correctness-only or
otherwise not scored for decode.

## This track

The ranked driver is `tools/bonsai2-9b-measure-and-score.sh`, which calls
`benchd iterate --mode official`. That path does **not** call measure-job.

Pool entries in `fixtures/bonsai2_9b_cuda_v1_track.json` therefore omit
`noop_decode_speedup` until an organizer measure-job workflow is added.

The fixture carries the rationale in `_noop_decode_speedup_note` beside
`timed_prompt_pool`.

## Action if measure-job is enabled later

For each pool golden, set `noop_decode_speedup` per benchd’s contract for
that prompt’s role (timed vs correctness-only). Re-run manifest lint and a
dry-ranked preflight after updating pins.
