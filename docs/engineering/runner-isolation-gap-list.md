# Runner isolation gap list: Bonsai 2 9B CUDA vs Eigen reference `m5-machine-scripts`

Read-only comparison (2026-10-08). **Eigen reference paths** are from
[Layr-Labs/m5-machine-scripts](https://github.com/Layr-Labs/m5-machine-scripts)
at commit `0783ba9` (for example `runner-isolation/supervisor.sh`).

**Live box facts** come from `[account-boundary-audit.md](account-boundary-audit.md)`
(audit Summary: `bonsai9b-box-runner` not installed; job identity is shared
`runner`).

**Naming correction:** Some early docs assumed a persistent Actions runner.
The **in-repo** supervisor is **JIT single-use per job**, same family as Reference 
(`tools/box-runner/supervisor.sh:5-8`, `357-365`). What is persistent on Sparks today is
the **host account** `runner` (Nemotron-era), not a separate `benchrunner` uid.

---

## 1. Side-by-side dimensions

| Dimension | OURS (repo + live audit where noted) | Eigen reference implementation (`m5-machine-scripts` runner-isolation) |
|-----------|--------------------------------------|------------------------------|
| **Job account: name** | `RUNNER_USER=runner` default (`tools/box-runner/box-runner.env.example:45`; `tools/box-runner/supervisor.sh:72`). Live: `runner` (`account-boundary-audit.md` Summary). | `benchrunner` (`Layr-Labs/m5-machine-scripts@0783ba9:boxes/test-spark.env:31-32`; `runner-isolation/isolation-config.sh:47-48`). |
| **Job account: uid/gid** | Not pinned in repo; live boxes use the shared `runner` account (`account-boundary-audit.md` Summary). | Pinned uid/gid in box env + `useradd` (`boxes/test-spark.env`; `runner-isolation/isolation-lib.sh`). |
| **Job account: groups** | Live: `runner` in **docker** for engine image builds (`account-boundary-audit.md` Summary). Repo assumes `runner` can use docker (`docs/sandbox-design.md:11-12`). | Primary **bench**; test-spark optional **docker** (`boxes/test-spark.env`; `runner-isolation/converge-isolation.sh`). |
| **Job account: sudo** | Live: passwordless sudo only into sandbox users (`account-boundary-audit.md` Summary). Not created by `install.sh` (`tools/box-runner/install.sh:4-16`). | **No sudo**; converge refuses admin/wheel/sudo membership (`runner-isolation/isolation-lib.sh`; `runner-isolation/janitor.sh`). |
| **Job account: shell** | `/bin/bash` implied (`setup-sandbox-user.sh` uses `su - runner` with login env: `tools/box-runner/setup-sandbox-user.sh:95-99`). Not set by box-runner install. | `/bin/bash` on Linux (`runner-isolation/isolation-lib.sh:165-168`). |
| **Job account: home** | Not pinned in repo; live boxes use the shared `runner` account home (`account-boundary-audit.md` Summary). `RUNNER_DIR` is the actions-runner install path (`tools/box-runner/box-runner.env.example:18-20`). | Operator-set home under box env (`boxes/test-spark.env`). |
| **Runner lifecycle** | JIT mint → one job → reap; stable **per-box** `RUNNER_NAME` for calibration (`tools/box-runner/supervisor.sh:141-149`, `352-357`; `tools/ranked-box-preflight.sh:746-749`). | JIT mint → one job → janitor; **unique** runner name per mint (`runner-isolation/supervisor.sh:80-87`, `81`). |
| **Token minting** | `gh-app-mint.sh` scoped token + revoke; JIT via App API (`tools/box-runner/gh-app-mint.sh:153-221`). | Same pattern; mint helper at `/opt/bench/gh-app-mint.sh` (`boxes/test-spark.env:43`; `runner-isolation/README.md:114-115`). |
| **Supervisor: user** | Root (no `User=`); template comment (`tools/box-runner/bonsai9b-box-runner.service.template:3-6`, `19-24`). | Root (`runner-isolation/units/systemd/benchrunner-supervisor.service:23-24`). |
| **Supervisor: unit** | `bonsai9b-box-runner.service` (rendered by `install.sh`; **not** auto-enabled) (`tools/box-runner/install.sh:165-190`). Live: unit **not installed** (`account-boundary-audit.md` Summary). | `benchrunner-supervisor.service` under `/etc/systemd/system` via `--apply` (`runner-isolation/converge-isolation.sh`). |
| **Supervisor: restart** | `Restart=on-failure`, `SuccessExitStatus=143`, `RestartSec=60` (`tools/box-runner/bonsai9b-box-runner.service.template:26-29`). | Same semantics; parks on quarantine **exit 0** (`runner-isolation/supervisor.sh:53-57`; `runner-isolation/units/systemd/benchrunner-supervisor.service:26-32`). |
| **Privilege drop** | `su - "${RUNNER_USER}"` + `ACTIONS_RUNNER_INPUT_JITCONFIG` via stdin file (`tools/box-runner/supervisor.sh:173-186`). | `su - benchrunner` + JIT in `RUNNER_DIR/.jitconfig.$$` owned by benchrunner (`runner-isolation/supervisor.sh:94-104`). |
| **Docker access** | Job account in **docker** group on live boxes (`account-boundary-audit.md` Summary). Job scripts call `docker` as `runner` (`tools/ranked-box-preflight.sh:545-555`; `tools/engine/build.sh:140`, `213`). | test-spark: **yes** by design (`boxes/test-spark.env`); janitor does **not** re-check docker membership. |
| **Sandbox / participant user** | **`bonsaisub`**: nologin, not docker/bench/sudo; `runner` sudo drop (`tools/box-runner/setup-sandbox-user.sh:198-203`, `300-308`, `327-336`; `docs/sandbox-design.md:219-225`). Present on audited bring-up boxes (`account-boundary-audit.md` Summary). | **None** in runner-isolation; job code runs as **benchrunner** (`runner-isolation/README.md`). |
| **Network egress (job)** | Not blocked for `runner` (threat model: participant code should not run as runner) (`docs/sandbox-design.md:11-18`). | Not specified in runner-isolation (benchrunner is the job identity). |
| **Network egress (sandbox)** | `iptables`/`ip6tables` uid-owner REJECT for `bonsaisub`; resolver ACLs (`tools/box-runner/setup-sandbox-user.sh:62-67`, `388-402`; `docs/sandbox-design.md:227-228`). | N/A (no parallel sandbox user). |
| **Filesystem: evaluator material** | Staged under **`/srv/bonsai9b/`**; refresh installs goldens `root:bench` `0640` (`tools/pool-refresh.sh:258`). Preflight requires the golden directory (`tools/ranked-box-preflight.sh:258-262`), pin-verifies bytes/sha (`264-274`, called at `280` and `313`) and refuses unpinned extra JSON files, except refresh-generation names matching `^g[0-9]+-[0-9]+(\.mtp[0-9]+)?\.golden\.json$` (`338-347`). Partial staging on bring-up boxes (`account-boundary-audit.md` Summary). | Under operator home + **`owner:bench` 0750 / 0640** trees listed in box env (`boxes/test-spark.env`; `runner-isolation/README.md`). |
| **Janitor / between jobs** | Supervisor: **pid-tree reap**, `remove_stray_residents` (`serve-up-*` containers only) (`tools/box-runner/supervisor.sh:230-265`, `367`). Workflow: **`sandbox-wipe.sh`** + kill `bonsaisub` (`tools/sandbox-wipe.sh:42-69`; `.github/workflows/benchmark.yml:361-382`, `676-713`). No quarantine flag. | **`janitor.sh`**: TERM/KILL all benchrunner procs, clear `_work` children, purge benchrunner `/tmp`; **boundary check** → quarantine (`runner-isolation/janitor.sh:9-14`, `93-130`, `48-57`). No docker/image scrub in janitor. |
| **Boundary re-check** | Preflight: **`setup-sandbox-user.sh --check`** only (`tools/ranked-box-preflight.sh:814-823`). Writable-tree probe for `runner` not in ranked preflight yet (`account-boundary-audit.md` Implications). | Janitor: no sudo + still in **bench** (`runner-isolation/janitor.sh`). **`access-probe.sh`**: staged assets readable, owner home denied, sudo denied (`runner-isolation/access-probe.sh`). |
| **GPU lock** | Workflow creates/holds **`/tmp/mtplx-gpu-exclusive.lock`** with `flock` (mode 0666 if created in job) (`.github/workflows/benchmark.yml:67-70`, `584-591`; `tools/calibrate-box.sh:89`). No tmpfiles unit in box-runner. | **`/tmp/mtplx-gpu-exclusive.lock`**: `root:bench` **0660**, tmpfiles at boot (`boxes/test-spark.env:52-53`; `runner-isolation/isolation-lib.sh:235-243`; `runner-isolation/converge-isolation.sh:59-60`, `214-215`). |
| **Secrets handling** | App key path via `BOX_RUNNER_APP_ENV` / `GH_APP_KEY`; not in `box-runner.env` (`tools/box-runner/box-runner.env.example:5-14`; `tools/box-runner/gh-app-mint.sh:45-50`). JIT off argv (`tools/box-runner/supervisor.sh:30-34`, `177-185`). | `/opt/bench/gh-app.env` + **`gh-app-key.pem` root 0600** (`runner-isolation/README.md:114-115`). Apply uses **`sudo -S`** password env (never logged) (`runner-isolation/README.md:120-128`; `runner-isolation/isolation-lib.sh:107-133`). |
| **Logs** | `LOG_DIR/supervisor.log`, per-job `runner-*.log`, systemd append stdout/stderr (`tools/box-runner/supervisor.sh:83-84`, `107-108`, `170-171`; `tools/box-runner/bonsai9b-box-runner.service.template:33-34`). | `ISO_DEPLOY_ROOT/log/supervisor.log`, `janitor.log`, runner logs (`runner-isolation/supervisor.sh:44-51`, `99`, `110`; unit `runner-isolation/units/systemd/benchrunner-supervisor.service:41-42`). |
| **Box description (env profile)** | Track fixture + env vars in workflow; **no** `boxes/<name>.env` in this repo. Calibration ties **`RUNNER_NAME`** to box (`tools/ranked-box-preflight.sh:746-749`). | **`boxes/<box>.env`** + `boxes/box-config.sh` (`boxes/README.md:1-17`; `runner-isolation/isolation-config.sh:4-8`). |
| **Converge (`--check` vs apply)** | **`install.sh`**: idempotent runner tarball + dirs + render unit; **no** `systemctl`, no account/GPU converge (`tools/box-runner/install.sh:12-16`, `179-194`). Sandbox: **`setup-sandbox-user.sh`** setup vs `--check` (`tools/box-runner/setup-sandbox-user.sh:574-595`). | **`converge-isolation.sh --check`** read-only report; **`--apply`** users, GPU lock, deploy payload, enable service (`runner-isolation/converge-isolation.sh:16-25`, `71-156`, `181-228`). |

---

## 2. GAP LIST (ranked)

Each item: what Reference has that we lack (or we conflict), why it matters, minimal change on our side, track conflicts.

### G1 — Between-job **janitor + quarantine** on the job account

- **Eigen reference:** `janitor.sh` after every job; drift → `quarantine.flag`; supervisor stops taking jobs (`runner-isolation/supervisor.sh:109-112`, `76-78`; `runner-isolation/janitor.sh:48-57`, `118-130`).
- **Ours:** Only `reap_between_jobs` + stray `serve-up-*` docker rm (`tools/box-runner/supervisor.sh:256-265`, `234-241`). No sudo/group drift check on **`runner`** between jobs.
- **Why:** A job or operator change to `runner` groups/sudo could persist across JIT cycles while the supervisor keeps minting jobs.
- **Minimal change:** Root `tools/box-runner/janitor.sh` (or extend supervisor) modeled on the Eigen reference: kill strays, optional `_work` top-level cleanup for `runner`, re-check “no docker” (if policy adopted) + no passwordless sudo except sandbox targets + still in expected groups; write quarantine; supervisor exits 0 when flagged (mirror `fatal_quarantine`).
- **Conflicts:** **docker group** is required today for `runner` (`docs/sandbox-design.md:11-12`; `account-boundary-audit.md` Summary and Implications §2). Janitor cannot enforce “no docker” until docker is removed or wrapped. **Shared `runner`** with Nemotron: janitor must not assume benchrunner-only uids.

### G2 — Fleet **box profile** + side-effect-free **`--check`** converge

- **Eigen reference:** `boxes/test-spark.env` + `converge-isolation.sh --check` (`boxes/README.md:19-29`; `runner-isolation/converge-isolation.sh:71-156`).
- **Ours:** Parameters scattered across `box-runner.env`, fixture, workflow; no single greppable host profile or `--check` for runner account + GPU lock + unit install state.
- **Why:** Hard to prove a Spark is in the intended posture before enabling ranked jobs; Spark box 3 staged, others not (`account-boundary-audit.md` Summary).
- **Minimal change:** Add `boxes/bonsai-<box>.env` (or document mapping each Spark box → `RUNNER_NAME`) and a thin `converge-box-runner.sh --check` that reports uid/groups, unit active, GPU lock mode, sandbox `--check`, staged path presence—without applying.
- **Conflicts:** None if check-only; **apply** must not fight other track units on shared Sparks (organizer ordering).

### G3 — **access-probe** / §7a writable-tree audit for the **job** account

- **Eigen reference:** `access-probe.sh` receipt (`runner-isolation/access-probe.sh:56-85`).
- **Ours:** Preflight runs **sandbox** `--check` only (`tools/ranked-box-preflight.sh:820-823`); writable-tree probe for **`runner`** not in ranked preflight yet (`account-boundary-audit.md` Implications §1; Follow-up).
- **Why:** `runner` can still be root-equivalent via docker while sandbox passes (`account-boundary-audit.md` Implications §2).
- **Minimal change:** Port Reference access-probe checks into `tools/ranked-box-preflight.sh` for official runs (paths from fixture/env); fail on docker membership if policy requires.
- **Conflicts:** **docker** + **shared runner** with Nemotron until helper/split account (`account-boundary-audit.md` Implications §2).

### G4 — **Pinned job uid** (`benchrunner`) vs shared **`runner`**

- **Eigen reference:** pinned uid in box env + `useradd` (`runner-isolation/isolation-lib.sh`; `boxes/test-spark.env`).
- **Ours:** Reuse existing **`runner`** (`tools/box-runner/box-runner.env.example`; `account-boundary-audit.md`).
- **Why:** Pinned uid simplifies egress rules and fleet maps; we already require distinct `bonsaisub` vs `nemosub` uids (`tools/box-runner/setup-sandbox-user.sh:207-210`).
- **Minimal change:** Either adopt `benchrunner@570` for Bonsai-only cutover (Reference path) or document why `runner` stays and encode each Spark box → uid in calibration `box` field.
- **Conflicts:** **High** — Nemotron track shares `runner` (`tools/box-runner/README.md:192-196`; `account-boundary-audit.md` Summary: `runner` in **docker** on live boxes).

### G5 — **GPU lock** ownership at boot (0660 `root:bench`) vs job-created 0666 file

- **Eigen reference:** tmpfiles + converge (`runner-isolation/isolation-lib.sh:235-243`; `runner-isolation/converge-isolation.sh:214-215`).
- **Ours:** Job may create lock with `umask 0000` (`.github/workflows/benchmark.yml:585-587`); design cites same path (`docs/design.md` via grep: `/tmp/mtplx-gpu-exclusive.lock`).
- **Why:** World-writable lock file weakens “bench group only” exclusivity if benchrunner model were adopted.
- **Minimal change:** One-shot tmpfiles or `setup-sandbox-user.sh`-class unit: `root:bench` 0660 before first job; workflow only opens existing file.
- **Conflicts:** Low if `bench` group already includes `runner` on Sparks (audit).

### G6 — **Supervisor parks on quarantine** (exit 0) vs only `Restart=on-failure`

- **Eigen reference:** `fatal_quarantine` exit 0 (`runner-isolation/supervisor.sh:53-57`).
- **Ours:** No quarantine concept (`tools/box-runner/supervisor.sh`—alerts on stop failure but no park-on-drift).
- **Why:** Prevents respawn loops while boundary is broken.
- **Minimal change:** Pair G1 with Reference  exit semantics.
- **Conflicts:** None.

### G7 — **`converge-isolation --apply`** installs and **enables** service

- **Eigen reference:** `systemctl enable --now` (`runner-isolation/converge-isolation.sh:217-218`).
- **Ours:** Operator manual steps (`tools/box-runner/install.sh:179-190`; `tools/box-runner/README.md:176-179`).
- **Why:** Documented split reduces accidental live enable; also means fleet drift (unit not installed on audit boxes).
- **Minimal change:** Optional `--apply` wrapper that copies rendered unit + enables (still root-only); or ops runbook only.
- **Conflicts:** Must order after `bonsai9b-sandbox-egress.service` (`tools/box-runner/bonsai9b-box-runner.service.template:16-17`).

### G8 — Reference supervisor: **no ExecStop / pid-tree** (weaker orphan handling)

- **Eigen reference:** Blocking `su` in foreground; `KillMode=process`; no `ExecStop` in shipped unit (`runner-isolation/units/systemd/benchrunner-supervisor.service:33-34`; `runner-isolation/supervisor.sh:103-105`).
- **Ours:** `PID_FILE`, `lib/pidtree.sh`, `ExecStop=supervisor.sh stop` (`tools/box-runner/supervisor.sh:19-28`, `267-301`; `tools/box-runner/bonsai9b-box-runner.service.template:24`, `9-12`).
- **Why:** This is **ours ahead of the Eigen reference** for `su -` session orphans—not a gap to close by adopting its supervisor verbatim.
- **Minimal change:** If aligning supervisors, **keep** pid-tree + ExecStop; port only janitor/quarantine/JIT file placement.
- **Conflicts:** N/A (prefer ours).

### G9 — **JIT runner naming**: stable per-box vs timestamp-per-mint

- **Eigen reference:** `RUNNER_NAME_PREFIX-$(date…)-$$` (`runner-isolation/supervisor.sh:81`).
- **Ours:** Stable `bonsai9b-box-${BOX_RUNNER_NAME\|hostname}` for **calibration `box` == RUNNER_NAME** (`tools/box-runner/supervisor.sh:352-357`; `tools/ranked-box-preflight.sh:746-749`).
- **Why:** Reference  names avoid GitHub clutter; ours **requires** stable name for preflight.
- **Minimal change:** Do **not** switch to Reference  naming without changing calibration contract.
- **Conflicts:** **Track requirement** — stable `RUNNER_NAME`.

### G10 — **Idle recycle**: background wait + grep vs `timeout` wrapping `run.sh`

- **Eigen reference:** `timeout ${IDLE_RECYCLE_SECONDS} ./run.sh` when `timeout` exists (`runner-isolation/supervisor.sh:100-103`).
- **Ours:** Background child + loop until idle recycle if no `Running job:` in log (`tools/box-runner/supervisor.sh:208-218`).
- **Why:** Ours allows `stop`/status while waiting; Reference blocks supervisor thread.
- **Minimal change:** Optional; keep ours unless simplifying.
- **Conflicts:** None.

### G11 — **Job account has sudo to sandbox** (ours) vs **no sudo at all** (Eigen reference)

- **Eigen reference:** benchrunner never sudo (`runner-isolation/README.md:19-21`; `access-probe.sh:80-85`).
- **Ours:** `runner` → `bonsaisub` (and live `nemosub`) (`tools/box-runner/setup-sandbox-user.sh:58-59`; `account-boundary-audit.md` Summary).
- **Why:** Required for `sandbox-wipe.sh`, `cargo`/`run-engine-sandboxed.sh` (`docs/sandbox-design.md:224-225`; `tools/sandbox-wipe.sh:40`).
- **Minimal change:** Do **not** remove sudo drop without replacing with root-owned setuid helpers for each sudo invocation.
- **Conflicts:** **Core Bonsai sandbox design** — not compatible with Reference  pure benchrunner model without a second user.

### G12 — **Docker in job account** (both test-spark configs) vs audit **7b extension**

- **Eigen reference:** Explicit `BOX_DOCKER_GROUP=yes` (`boxes/test-spark.env:56-59`).
- **Ours:** Same on live `runner`; scripts require docker CLI (`tools/ranked-box-preflight.sh:545-545`).
- **Why:** Root-equivalent unless constrained helper (`account-boundary-audit.md` Implications §2).
- **Minimal change:** `bonsai9b-docker` helper + remove `runner` from docker (audit Follow-up).
- **Conflicts:** **Nemotron** on same account; kernels container lacks `--read-only` today (repo gap, not covered by the audit).

### G13 — **Staged asset layout**: `/srv/bonsai9b` vs operator-home trees

- **Eigen reference:** Consumes `owner:bench` under `BOX_STAGED_DIRS` (`runner-isolation/converge-isolation.sh:115-126`).
- **Ours:** `/srv/bonsai9b/*` + preflight pins (`tools/ranked-box-preflight.sh:446-458`, `547-553`).
- **Why:** Different staging lane; access-probe paths won’t map 1:1.
- **Minimal change:** Bonsai-specific `access-probe` paths (goldens, benchd-bin, baseline, calibration file, weights, engine-image.id) under `/srv/bonsai9b/` (`account-boundary-audit.md` Summary; Implications §1).
- **Conflicts:** None if paths stay `/srv/bonsai9b`.

### G14 — **Production hold / `--production-scheduled`**

- **Eigen reference:** Refuses apply on held boxes (`runner-isolation/converge-isolation.sh:183-185`; `boxes/test-spark.env:61`).
- **Ours:** No equivalent in box-runner.
- **Why:** Prevents accidental cutover on shared production hosts.
- **Minimal change:** Flag in box env if we add converge apply.
- **Conflicts:** Sparks run multiple tracks—operational not code.

### G15 — **Requires sandbox egress unit before runner** (ours only)

- **Ours:** `Requires=bonsai9b-sandbox-egress.service` + iptables ExecStartPre (`tools/box-runner/bonsai9b-box-runner.service.template:16-22`).
- **Eigen reference:** No analogous Requires in `benchrunner-supervisor.service` (`runner-isolation/units/systemd/benchrunner-supervisor.service:14-19`).
- **Why:** Fail-closed if `bonsaisub` egress rules missing.
- **Minimal change:** Keep when adopting Reference patterns.
- **Conflicts:** None.

---

## 3. In OURS that Reference  runner-isolation does **not** have (preserve if merging)

| Capability | Where |
|------------|--------|
| **`bonsaisub` participant sandbox** (second uid, sudo drop, iptables egress, resolver ACLs, wipe list, cron/at deny) | `tools/box-runner/setup-sandbox-user.sh`; `docs/sandbox-design.md` §5–6a |
| **Workflow-integrated `sandbox-wipe.sh`** (kill + wipe + SysV IPC) at job start/end and measure paths | `tools/sandbox-wipe.sh`; `.github/workflows/benchmark.yml:361-382`, `676-713`; `docs/sandbox-design.md:186-201` |
| **Container hardening for resident/build** (`--network none`, `--read-only`, cap-drop, `bonsaisub` in container) | `tools/serve-up.sh:548-551`; `tools/engine/build.sh:133-140`, `205-211` |
| **`lib/pidtree.sh` + `PID_FILE` + `ExecStop` orphan recovery** | `tools/box-runner/supervisor.sh:19-28`, `201-301`; `tools/box-runner/bonsai9b-box-runner.service.template:9-12`, `24` |
| **Stray GPU resident cleanup** (`docker rm` `serve-up-*`) | `tools/box-runner/supervisor.sh:230-241` |
| **Stable JIT runner name ↔ calibration `box`** | `tools/box-runner/supervisor.sh:352-357`; `tools/ranked-box-preflight.sh:746-749` |
| **In-repo `gh-app-mint.sh` with stale offline runner cleanup** | `tools/box-runner/gh-app-mint.sh:191-212` |
| **Checksum-pinned `actions-runner` install** | `tools/box-runner/install.sh:42-127` |
| **Parallel Nemotron sandbox coexistence** (`nemosub`, shared `/opt/rust`) | `tools/box-runner/setup-sandbox-user.sh:207-210`, `351-361`; `tools/box-runner/README.md:192-205` |
| **Ranked preflight** (fixture, engine image id, sandbox check, GPU driver, calibration band) | `tools/ranked-box-preflight.sh` (e.g. `814-825`, `746-749`) |

---

## 4. Speculation / open points

- Whether any Spark still uses a **persistent PAT-registered** Actions runner for Nemotron was **not** verified in this pass.
- Reference  **converge/** staging lane for CUDA boxes was not line-compared here; isolation explicitly **does not chown** staged trees (`runner-isolation/converge-isolation.sh:12-14`, `207-208`). Bonsai staging is implied via `/srv/bonsai9b` + organizer tooling in preflight comments (`tools/ranked-box-preflight.sh:446-458`).
