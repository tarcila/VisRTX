# VisRTX — Local Workspace Rig

**This file is personal, local-only scaffolding — not part of the VisRTX source tree.**

The repo root is a common-to-all-workspaces setup. It holds only what's specific to *this machine's* layout: the shared `Justfile`, the Nix dev shell, and the jj workspaces (as subfolders). It does **not** contain the source tree.

Authoritative, code-facing guidance lives **in each workspace**:

- `<workspace>/AGENTS.md` — project / device / TSD architecture
- `<workspace>/devices/rtx/AGENTS.md`, `<workspace>/devices/gl/AGENTS.md`, `<workspace>/tsd/AGENTS.md` — per-component details
- `<workspace>/STYLEGUIDE.md` — C++/CUDA coding conventions

Don't restate here what those cover. This file stays lean.

## Scratch: handoffs, specs, tickets

Handoff documents, specs, and ticket/tracking documents **always** go in a
`.scratch/` subfolder (workspace-local). Never `/tmp`, never the source tree.

**Never push these to GitHub or open issues/PRs unless explicitly instructed.**
Local-first: these are working notes, not published artifacts.

## Version Control: Jujutsu

This project uses **Jujutsu (jj)**. Never use raw `git` in this repo.

Key concepts:
- Working copy is tracked automatically (no staging)
- Changes are immutable once published — use `jj new` to start fresh work
- `jj describe` sets commit message; `jj log` shows history

### Workspaces

Each workspace is an independent checkout at a different revision, sharing one underlying store. They sit as **sibling subfolders under this repo root**, each a full source checkout with its own build dir — so RTX / GL / TSD work runs in parallel without context switching.

`next_release` is the long-lived base workspace (develop on top of it unless directed otherwise). The rest are ephemeral: created per feature, dropped when merged. Name them after the work focus, bare and hyphen-separated — `emissive-geometry`, `volume-sampler`, `fix-triangle-normals` — no prefix.

```bash
jj workspace list                       # list all workspaces
just new_workspace <name> <tracking>    # new workspace subfolder tracking a remote branch
just forget_workspace <name>
cd ../<workspace>                        # switch focus
```

### Common Commands

```bash
jj status               # working copy status
jj diff                 # uncommitted changes
jj log                  # graphical history
jj new                  # new change on top of current
jj edit <rev>           # edit existing change (working copy IS that rev)
jj squash               # squash into parent
jj rebase -d next_release
jj abandon              # discard current change
```

### Feature Development Loop

1. Build after each change: `just build`
2. Commit with a terse message: `jj commit -m "Add foo support"`
3. To iterate on an existing commit: `jj edit <rev>` → fix → `just build` → `jj new <tip>` to return

Keeps history clean: each commit is a complete, working change.

## Build (via `just`)

`Justfile` lives at this repo root and is shared across workspaces. It drives a **Ninja Multi-Config** build and derives `VISRTX_LIBRARY_NAME` per-workspace (from the subfolder name), so each workspace's device library is uniquely named.

```bash
just prepare            # generate .clangd, .envrc, VS Code, Emacs configs
just configure          # MDL on, GL off, TSD on, tests on
just configure_mdl_off
just configure_physicallybased_mdl
just build              # build RelWithDebInfo (default)
just br | brd | bd      # Release / RelWithDebInfo / Debug
just rebuild            # clean + build
just view  *ARGS        # run tsdViewer
just render *ARGS       # run tsdRender
just clean              # remove _out/
```

Run `just --list` for the current set; this list may drift.

Build dir: `_out/_cmake/`. Install dir: `_out/`. Binaries: `_out/bin/`.

RTX API tests load the **installed** `.so`, not the build-tree one — install before running or the test exercises stale code. `just brd` builds *and* installs; run it before every test pass.

For the full CMake option set, coding conventions, and CI, see the in-tree `<workspace>/AGENTS.md` and `<workspace>/STYLEGUIDE.md`.

## Environment Setup

Nix flake provides the dev shell:

```bash
nix develop        # or: direnv allow
```

The shell wires up multiple ANARI devices (visrtx, barney, helide, cycles, ospray, visionaray; visgl on x86_64 Linux). For env vars (`TSD_ANARI_LIBRARIES`, `ANARI_LIBRARY`, …), `just prepare` writes an `.envrc` pinning this workspace's library name.
