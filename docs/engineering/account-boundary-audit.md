# Account-boundary audit (contest Sparks)

Audit date: 2026-10-08. Scope: ranked job identity, sandbox user, and staged
evaluator paths on DGX Sparks used for Bonsai bring-up.

## Summary

- **`bonsai9b-box-runner.service` was not installed** on audited boxes; ranked
  jobs still used the shared `runner` account from the prior track.
- **Sandbox user `bonsaisub`** was present (example uid 995 on Spark box 3) with
  nologin shell and egress rules via `bonsai9b-sandbox-egress`.
- **Job account `runner`** was in the **docker** group on live boxes; ranked
  scripts invoke `docker` as `runner` for engine image builds and residents.
- **Sudo for `runner`** was limited to dropping into sandbox users (`bonsaisub`,
  and legacy `nemosub` on shared hosts); passwordless `sudo -n true` as `runner`
  itself failed (expected).
- **Staged evaluator material** for Bonsai lived under `/srv/bonsai9b/` on
  Spark box 3 (goldens, weights path, baseline tree references); other Sparks were
  not fully staged at audit time.

## Implications

1. Preflight enforces sandbox `--check` but not a full “job account writable
   tree” probe equivalent to Eigen’s access-probe (see
   [runner-isolation-gap-list.md](runner-isolation-gap-list.md)).
2. Docker membership on `runner` is root-equivalent unless wrapped; removing
   it requires a helper or split account without breaking Nemotron coexistence
   on the same host.
3. Installing `bonsai9b-box-runner` and enabling the unit remains an organizer
   cutover step after staging and calibration.

## Follow-up (organizer)

- Port §7a/7b writable-tree checks into `tools/ranked-box-preflight.sh` when
  policy requires “no docker on job account”.
- Decide per host: shared `runner` vs dedicated `benchrunner` uid for Bonsai-only
  cutover.
